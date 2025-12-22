#!/usr/bin/env python
"""
Azure ML Pipeline Step 2: Neo4J Preparation
Processes visitor, session, job stream, specialization stream, and relationship data for Neo4j database.
"""

import os
import sys
import json
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import pandas as pd
import mlflow
from dotenv import load_dotenv

# Azure ML imports
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azureml.fsspec import AzureMachineLearningFileSystem

# Add project root to path for PA imports
root_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(root_dir)
sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
pa_dir = os.path.join(parent_dir, 'PA')
# Add paths
sys.path.insert(0, parent_dir)
sys.path.insert(0, pa_dir)

# Import PA processors
from PA.neo4j_visitor_processor import Neo4jVisitorProcessor
from PA.neo4j_session_processor import Neo4jSessionProcessor
from PA.neo4j_job_stream_processor import Neo4jJobStreamProcessor
from PA.neo4j_specialization_stream_processor import Neo4jSpecializationStreamProcessor
from PA.neo4j_visitor_relationship_processor import Neo4jVisitorRelationshipProcessor
from PA.utils.config_utils import load_config
from PA.utils.logging_utils import setup_logging
from PA.utils.keyvault_utils import ensure_env_file, KeyVaultManager
from PA.utils.app_insights import configure_app_insights
from neo4j_env_utils import apply_neo4j_credentials
from step_input_sync import stage_step1_outputs


class Neo4jPreparationStep:
    """Azure ML Neo4j Preparation Step for Personal Agendas pipeline."""
    
    def __init__(
        self,
        config_path: str,
        incremental: bool = False,
        use_keyvault: bool = True,
        neo4j_environment: Optional[str] = None,
    ):
        """
        Initialize the Neo4j Preparation Step.
        
        Args:
            config_path: Path to configuration file
            incremental: Whether to run incremental processing
            use_keyvault: Whether to use Azure Key Vault for secrets
        """
        self.config_path = config_path
        self.incremental = incremental
        self.use_keyvault = use_keyvault
        self.neo4j_environment_override = neo4j_environment
        self.logger = self._setup_logging()
        
        self.config = self._load_configuration(config_path)
        self.selected_neo4j_environment = self._apply_environment_override()

        # Load secrets from Key Vault if in Azure ML
        if self.use_keyvault and self._is_azure_ml_environment():
            self._load_secrets_from_keyvault()
        # IMPORTANT: For Neo4j processors, create_only_new=True means incremental mode
        # This is opposite of the incremental flag logic, so we use incremental directly
        self.create_only_new = self.incremental
    
    def _is_azure_ml_environment(self) -> bool:
        """Check if running in Azure ML environment."""
        return any([
            os.environ.get("AZUREML_RUN_ID"),
            os.environ.get("AZUREML_EXPERIMENT_NAME"),
            os.environ.get("AZUREML_WORKSPACE_NAME")
        ])
    
    def _load_secrets_from_keyvault(self) -> None:
        """Load secrets from Azure Key Vault."""
        try:
            keyvault_name = os.environ.get("KEYVAULT_NAME", "strategicai-kv-uks-dev")
            self.logger.info(f"Loading secrets from Key Vault: {keyvault_name}")
            
            # Try multiple approaches to get credentials
            # First, check if they're already in environment variables
            neo4j_uri = os.environ.get("NEO4J_URI")
            neo4j_username = os.environ.get("NEO4J_USERNAME")
            neo4j_password = os.environ.get("NEO4J_PASSWORD")
            
            if all([neo4j_uri, neo4j_username, neo4j_password]):
                apply_neo4j_credentials(
                    self.config,
                    neo4j_uri,
                    neo4j_username,
                    neo4j_password,
                    logger=self.logger,
                    environment_override=self.selected_neo4j_environment,
                )
                return

            if not all([neo4j_uri, neo4j_username, neo4j_password]):
                # Try to get from Key Vault
                try:
                    kv_manager = KeyVaultManager(keyvault_name)
                    
                    # Get Neo4j specific secrets
                    neo4j_secrets = {
                        "NEO4J_URI": kv_manager.get_secret("neo4j-uri") or kv_manager.get_secret("NEO4J-URI"),
                        "NEO4J_USERNAME": kv_manager.get_secret("neo4j-username") or kv_manager.get_secret("NEO4J-USERNAME"),
                        "NEO4J_PASSWORD": kv_manager.get_secret("neo4j-password") or kv_manager.get_secret("NEO4J-PASSWORD")
                    }
                    
                    # Set environment variables for Neo4j connection
                    for key, value in neo4j_secrets.items():
                        if value:
                            os.environ[key] = value
                            self.logger.info(f"Loaded {key} from Key Vault")
                except Exception as kv_error:
                    self.logger.warning(f"Could not load from Key Vault: {kv_error}")
            
            apply_neo4j_credentials(
                self.config,
                os.environ.get("NEO4J_URI"),
                os.environ.get("NEO4J_USERNAME"),
                os.environ.get("NEO4J_PASSWORD"),
                logger=self.logger,
                environment_override=self.selected_neo4j_environment,
            )
            
        except Exception as e:
            self.logger.error(f"Error in Key Vault setup: {e}")
            self.logger.info("Neo4j credentials must be provided via environment variables or .env file")
    
    def _apply_environment_override(self) -> Optional[str]:
        """Ensure the config reflects the highest-precedence Neo4j environment."""
        config_env = (self.config.get('neo4j', {}) or {}).get('environment')
        selected = self.neo4j_environment_override or config_env

        if selected:
            normalized = str(selected).strip()
            self.config.setdefault('neo4j', {})['environment'] = normalized
            self.logger.info("Using Neo4j environment '%s'", normalized)
            return normalized

        self.logger.info("Neo4j environment not specified; default resolution will apply")
        return None

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(root_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'neo4j_prep_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _load_configuration(self, config_path: str) -> Dict[str, Any]:
        """
        Load and validate configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        self.logger.info(f"Loading configuration from: {config_path}")
        
        # Ensure config path is absolute
        if not os.path.isabs(config_path):
            config_path = os.path.join(project_root, config_path)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        config = load_config(config_path)
        
        # Update config for incremental mode if needed
        if self.incremental:
            config['incremental'] = True
            self.logger.info("Running in incremental mode")
        
        return config
    
    def setup_data_directories(self, base_dir: str) -> Dict[str, str]:
        """
        Setup data directories for processing.
        
        Args:
            base_dir: Base directory for data
            
        Returns:
            Dictionary of directory paths
        """
        directories = {
            'data': os.path.join(base_dir, 'data'),
            'output': os.path.join(base_dir, 'data', 'output'),
            'logs': os.path.join(base_dir, 'logs')
        }
        
        for name, path in directories.items():
            os.makedirs(path, exist_ok=True)
            self.logger.info(f"Created/verified directory: {name} -> {path}")
        
        return directories

    def _normalize_config_path(self, path_value: str) -> str:
        text = (path_value or '').replace('\\', '/').strip()
        while text.startswith('./'):
            text = text[2:]
        return text.lstrip('/')

    def _relative_support_path(self, path_value: str, event_name: str) -> str:
        normalized = self._normalize_config_path(path_value)
        event_prefix = f"data/{event_name}/"
        if normalized.lower().startswith(event_prefix.lower()):
            return normalized[len(event_prefix):]
        return os.path.basename(normalized)

    def _get_neo4j_support_targets(self, event_name: str) -> Dict[str, str]:
        targets: Dict[str, str] = {}
        neo4j_cfg = self.config.get('neo4j', {}) or {}

        mapping_specs = [
            (neo4j_cfg.get('job_stream_mapping', {}) or {}).get('file'),
            (neo4j_cfg.get('specialization_stream_mapping', {}) or {}).get('file'),
        ]

        for path_value in mapping_specs:
            if not path_value:
                continue
            relative_path = self._relative_support_path(path_value, event_name)
            if not relative_path:
                continue
            targets.setdefault(os.path.basename(relative_path).lower(), relative_path)

        return targets
    
    def copy_input_data(self, input_paths: Dict[str, str], data_dir: str) -> bool:
        """
        Copy input data from Step 1 outputs to expected local directory structure.
        This recreates the folder structure that PA processors expect based on config files.
        
        Args:
            input_paths: Dictionary of input paths from argparse
            data_dir: Local data directory (usually 'data')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            support_targets = self._get_neo4j_support_targets(
                self.config.get('event', {}).get('name', 'ecomm')
            )
            expected_files = {
                # Registration outputs
                'df_reg_demo_this.csv': ['df_reg_demo_this.csv'],
                'df_reg_demo_last_bva.csv': ['df_reg_demo_last_bva.csv'],
                'df_reg_demo_last_lva.csv': ['df_reg_demo_last_lva.csv'],
                'registration_data_with_demographicdata_bva_this.csv': ['Registration_data_with_demographicdata_bva_this.csv'],
                'registration_data_with_demographicdata_bva_last.csv': ['Registration_data_with_demographicdata_bva_last.csv'],
                'registration_data_with_demographicdata_lva_last.csv': ['Registration_data_with_demographicdata_lva_last.csv'],
                'registration_data_with_demographicdata_lva_this.csv': ['Registration_data_with_demographicdata_lva_this.csv'],

                # Scan outputs
                'sessions_visited_last_bva.csv': ['sessions_visited_last_bva.csv'],
                'sessions_visited_last_lva.csv': ['sessions_visited_last_lva.csv'],
                'scan_bva_past.csv': ['scan_bva_past.csv'],
                'scan_lva_past.csv': ['scan_lva_past.csv'],

                # Session outputs
                'session_this_filtered_valid_cols.csv': ['session_this_filtered_valid_cols.csv'],
                'session_last_filtered_valid_cols_bva.csv': ['session_last_filtered_valid_cols_bva.csv'],
                'session_last_filtered_valid_cols_lva.csv': ['session_last_filtered_valid_cols_lva.csv'],
                'streams.json': ['streams.json'],
                'streams_cache.json': ['streams_cache.json'],
                'job_to_stream.csv': ['job_to_stream.csv'],
                'spezialization_to_stream.csv': ['spezialization_to_stream.csv'],
                'teatres.csv': ['teatres.csv'],
                'bva25_session_export.csv': ['BVA25_session_export.csv'],
                'lvs24_session_export.csv': ['LVS24_session_export.csv'],
                'lvs25_session_export.csv': ['LVS25_session_export.csv'],
            }

            result = stage_step1_outputs(
                self.config,
                input_paths,
                data_dir,
                self.logger,
                support_targets=support_targets,
                expected_files=expected_files,
            )

            os.chdir(root_dir)
            self.logger.info("Changed working directory to: %s", root_dir)

            return bool(result.copied_files)

        except Exception as e:
            self.logger.error(f"Error copying input data: {e}")
            traceback.print_exc()
            return False
    
    def run_neo4j_visitor_processing(self) -> Dict[str, Any]:
        """Run Neo4j visitor data processing."""
        try:
            self.logger.info("Initializing Neo4j visitor processor")
            processor = Neo4jVisitorProcessor(self.config)
            processor.process(create_only_new=self.create_only_new)
            
            result = {
                'status': 'success',
                'statistics': processor.statistics if hasattr(processor, 'statistics') else {}
            }
            
            self.logger.info(f"Neo4j visitor processing completed: {result['statistics']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Neo4j visitor processing: {e}")
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def run_neo4j_session_processing(self) -> Dict[str, Any]:
        """Run Neo4j session data processing."""
        try:
            self.logger.info("Initializing Neo4j session processor")
            processor = Neo4jSessionProcessor(self.config)
            processor.process(create_only_new=self.create_only_new)
            
            result = {
                'status': 'success',
                'statistics': processor.statistics if hasattr(processor, 'statistics') else {}
            }
            
            self.logger.info(f"Neo4j session processing completed: {result['statistics']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Neo4j session processing: {e}")
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def run_neo4j_job_stream_processing(self) -> Dict[str, Any]:
        """Run Neo4j job to stream relationship processing."""
        try:
            self.logger.info("Initializing Neo4j job stream processor")
            processor = Neo4jJobStreamProcessor(self.config)
            processor.process(create_only_new=self.create_only_new)
            
            result = {
                'status': 'success',
                'statistics': processor.statistics if hasattr(processor, 'statistics') else {}
            }
            
            self.logger.info(f"Neo4j job stream processing completed: {result['statistics']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Neo4j job stream processing: {e}")
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def run_neo4j_specialization_stream_processing(self) -> Dict[str, Any]:
        """Run Neo4j specialization to stream relationship processing."""
        try:
            self.logger.info("Initializing Neo4j specialization stream processor")
            processor = Neo4jSpecializationStreamProcessor(self.config)
            processor.process(create_only_new=self.create_only_new)
            
            result = {
                'status': 'success',
                'statistics': processor.statistics if hasattr(processor, 'statistics') else {}
            }
            
            self.logger.info(f"Neo4j specialization stream processing completed: {result['statistics']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Neo4j specialization stream processing: {e}")
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def run_neo4j_visitor_relationship_processing(self) -> Dict[str, Any]:
        """Run Neo4j visitor relationship processing."""
        try:
            self.logger.info("Initializing Neo4j visitor relationship processor")
            processor = Neo4jVisitorRelationshipProcessor(self.config)
            processor.process(create_only_new=self.create_only_new)
            
            result = {
                'status': 'success',
                'statistics': processor.statistics if hasattr(processor, 'statistics') else {}
            }
            
            self.logger.info(f"Neo4j visitor relationship processing completed: {result['statistics']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Neo4j visitor relationship processing: {e}")
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def save_outputs(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save processing outputs to Azure ML path.
        
        Args:
            results: Processing results from all processors
            output_path: Output path from argparse
        """
        self.logger.info("Saving outputs to Azure ML path")
        
        if output_path:
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Create summary metadata file
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config_path,
                'incremental': self.incremental,
                'processors_run': list(results.keys()),
                'results': {}
            }
            
            # Add statistics from each processor
            for processor_name, processor_results in results.items():
                metadata['results'][processor_name] = {
                    'status': processor_results.get('status', 'unknown'),
                    'statistics': processor_results.get('statistics', {})
                }
            
            # Save metadata file
            metadata_file = os.path.join(output_path, 'neo4j_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved metadata to {metadata_file}")
            
            # Save statistics summary
            stats_file = os.path.join(output_path, 'neo4j_statistics.txt')
            with open(stats_file, 'w') as f:
                f.write("Neo4j Processing Statistics\n")
                f.write("=" * 60 + "\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Config: {self.config_path}\n")
                f.write(f"Incremental: {self.incremental}\n")
                f.write("\n")
                
                for processor_name, processor_results in results.items():
                    f.write(f"\n{processor_name}:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Status: {processor_results.get('status', 'unknown')}\n")
                    
                    stats = processor_results.get('statistics', {})
                    if stats:
                        if 'nodes_created' in stats:
                            f.write(f"Nodes created: {sum(stats['nodes_created'].values())}\n")
                        if 'nodes_skipped' in stats:
                            f.write(f"Nodes skipped: {sum(stats['nodes_skipped'].values())}\n")
                        if 'relationships_created' in stats:
                            if isinstance(stats['relationships_created'], dict):
                                f.write(f"Relationships created: {sum(stats['relationships_created'].values())}\n")
                            else:
                                f.write(f"Relationships created: {stats['relationships_created']}\n")
                        if 'relationships_skipped' in stats:
                            if isinstance(stats['relationships_skipped'], dict):
                                f.write(f"Relationships skipped: {sum(stats['relationships_skipped'].values())}\n")
                            else:
                                f.write(f"Relationships skipped: {stats['relationships_skipped']}\n")
            
            self.logger.info(f"Saved statistics to {stats_file}")
    
    def process(self) -> Dict[str, Any]:
        """
        Run the complete Neo4j preparation step.
        
        Returns:
            Dictionary containing results from all processors
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Neo4j Preparation Step")
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Incremental: {self.incremental}")
        self.logger.info("=" * 60)
        
        # Setup directories
        directories = self.setup_data_directories(root_dir)
        
        # Initialize results
        results = {}
        
        # Run processors based on configuration
        processors_config = self.config.get('processors', {})
        
        # Neo4j visitor processing (Step 4)
        if processors_config.get('neo4j_visitor_processing', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 40)
            self.logger.info("NEO4J VISITOR PROCESSING")
            self.logger.info("=" * 40)
            results['neo4j_visitor'] = self.run_neo4j_visitor_processing()
        
        # Neo4j session processing (Step 5)
        if processors_config.get('neo4j_session_processing', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 40)
            self.logger.info("NEO4J SESSION PROCESSING")
            self.logger.info("=" * 40)
            results['neo4j_session'] = self.run_neo4j_session_processing()
        
        # Neo4j job stream processing (Step 6)
        if processors_config.get('neo4j_job_stream_processing', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 40)
            self.logger.info("NEO4J JOB STREAM PROCESSING")
            self.logger.info("=" * 40)
            results['neo4j_job_stream'] = self.run_neo4j_job_stream_processing()
        
        # Neo4j specialization stream processing (Step 7)
        if processors_config.get('neo4j_specialization_stream_processing', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 40)
            self.logger.info("NEO4J SPECIALIZATION STREAM PROCESSING")
            self.logger.info("=" * 40)
            results['neo4j_specialization_stream'] = self.run_neo4j_specialization_stream_processing()
        
        # Neo4j visitor relationship processing (Step 8)
        if processors_config.get('neo4j_visitor_relationship_processing', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 40)
            self.logger.info("NEO4J VISITOR RELATIONSHIP PROCESSING")
            self.logger.info("=" * 40)
            results['neo4j_visitor_relationship'] = self.run_neo4j_visitor_relationship_processing()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Neo4j Preparation Step Completed")
        self.logger.info("=" * 60)
        
        return results


def main(args):
    """Main entry point for Azure ML step."""
    print("\n" + "=" * 60)
    print("AZURE ML NEO4J PREPARATION STEP")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    configure_app_insights(service_name="pa_step2_neo4j_prep")
    
    try:
        # Initialize the step
        step = Neo4jPreparationStep(
            args.config,
            incremental=args.incremental,
            neo4j_environment=args.neo4j_environment,
        )
        
        # Copy input data from Step 1 outputs if provided
        input_paths = {
            'input_registration': args.input_registration,
            'input_scan': args.input_scan,
            'input_session': args.input_session
        }
        
        # Check if we have input paths
        if any(input_paths.values()):
            data_dir = os.path.join(root_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Copy data from Step 1 outputs
            if not step.copy_input_data(input_paths, data_dir):
                print("Warning: Could not copy all input data from Step 1")
        
        # Run Neo4j preparation step
        results = step.process()
        
        # Save outputs to Azure ML location
        step.save_outputs(results, args.output_metadata)
        
        print("\n" + "=" * 60)
        print("NEO4J PREPARATION STEP SUMMARY")
        print("=" * 60)
        print(f"Configuration: {args.config}")
        print(f"Incremental: {args.incremental}")
        if args.neo4j_environment:
            print(f"Neo4j environment override: {args.neo4j_environment}")
        print(f"Results:")
        
        for processor, result in results.items():
            status = result.get('status', 'unknown')
            print(f"  - {processor}: {status}")
            
            # Print statistics if available
            stats = result.get('statistics', {})
            if stats:
                if 'nodes_created' in stats:
                    total_created = sum(stats['nodes_created'].values())
                    print(f"    Nodes created: {total_created}")
                if 'relationships_created' in stats:
                    if isinstance(stats['relationships_created'], dict):
                        total_rels = sum(stats['relationships_created'].values())
                    else:
                        total_rels = stats['relationships_created']
                    print(f"    Relationships created: {total_rels}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"JOB FAILED: {str(e)}")
        traceback.print_exc()
        raise


def _bool_arg(value: Optional[str]) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"true", "1", "yes", "y"}:
        return True
    if value_str in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected for incremental flag")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Azure ML Neo4j Preparation Step')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file (e.g., PA/config/config_ecomm.yaml)'
    )
    
    parser.add_argument(
        '--incremental',
        nargs='?',
        const=True,
        default=False,
        type=_bool_arg,
        help='Run in incremental mode (accepts true/false)'
    )

    parser.add_argument(
        '--neo4j_environment',
        type=str,
        help='Override Neo4j environment (dev|test|prod)'
    )
    
    # Input paths from Step 1
    parser.add_argument(
        '--input_registration',
        type=str,
        help='Path to registration data from Step 1'
    )
    
    parser.add_argument(
        '--input_scan',
        type=str,
        help='Path to scan data from Step 1'
    )
    
    parser.add_argument(
        '--input_session',
        type=str,
        help='Path to session data from Step 1'
    )
    
    # Output path
    parser.add_argument(
        '--output_metadata',
        type=str,
        help='Path to save metadata and statistics'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)