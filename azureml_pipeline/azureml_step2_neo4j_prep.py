#!/usr/bin/env python
"""
Azure ML Pipeline Step 2: Neo4J Preparation
Processes visitor, session, job stream, specialization stream, and relationship data for Neo4j database.
"""

import os
import sys
import json
import shutil
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


class Neo4jPreparationStep:
    """Azure ML Neo4j Preparation Step for Personal Agendas pipeline."""
    
    def __init__(self, config_path: str, incremental: bool = False, use_keyvault: bool = True):
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
        self.logger = self._setup_logging()
        
        # Load secrets from Key Vault if in Azure ML
        if self.use_keyvault and self._is_azure_ml_environment():
            self._load_secrets_from_keyvault()
        
        self.config = self._load_configuration(config_path)
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
            
            # Create or update .env file with available credentials
            env_path = os.path.join(project_root, "PA", "keys", ".env")
            os.makedirs(os.path.dirname(env_path), exist_ok=True)
            
            # Read existing .env file if it exists
            existing_env = {}
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        if '=' in line and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            existing_env[key] = value
            
            # Update with Neo4j credentials from environment
            if os.environ.get("NEO4J_URI"):
                existing_env["NEO4J_URI"] = os.environ.get("NEO4J_URI")
            if os.environ.get("NEO4J_USERNAME"):
                existing_env["NEO4J_USERNAME"] = os.environ.get("NEO4J_USERNAME")
            if os.environ.get("NEO4J_PASSWORD"):
                existing_env["NEO4J_PASSWORD"] = os.environ.get("NEO4J_PASSWORD")
            
            # Write updated .env file
            with open(env_path, 'w') as f:
                f.write("# Neo4j credentials\n")
                for key, value in existing_env.items():
                    f.write(f"{key}={value}\n")
            
            self.logger.info(f"Updated .env file at {env_path}")
            
            # Ensure the file can be read by the PA processors
            load_dotenv(env_path, override=True)
            
        except Exception as e:
            self.logger.error(f"Error in Key Vault setup: {e}")
            self.logger.info("Neo4j credentials must be provided via environment variables or .env file")
    
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
            # Get event name from config
            event_name = self.config.get('event', {}).get('name', 'ecomm')
            
            # Create the expected directory structure
            # Neo4j processors look in data/ecomm/output/
            output_dir = os.path.join(data_dir, event_name, 'output')
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Created output directory: {output_dir}")
            
            # Also create the data/output directory as Step 1 may have created files there
            alt_output_dir = os.path.join(data_dir, 'output')
            os.makedirs(alt_output_dir, exist_ok=True)
            
            # Track all copied files
            copied_files = []
            
            # Debug: List what's in each input path
            for input_key in ['input_registration', 'input_scan', 'input_session']:
                if input_key in input_paths and input_paths[input_key]:
                    input_path = input_paths[input_key]
                    self.logger.info(f"Checking {input_key} at: {input_path}")
                    
                    if os.path.exists(input_path) and os.path.isdir(input_path):
                        files = os.listdir(input_path)
                        self.logger.info(f"  Found {len(files)} files: {files}")
                    else:
                        self.logger.warning(f"  Path does not exist or is not a directory")
            
            # Copy from Step 1 output directories
            # These are the files Step 1 should produce based on the processors
            expected_files = {
                # Registration processor outputs (in data/output/)
                'df_reg_demo_this.csv': 'df_reg_demo_this.csv',
                'df_reg_demo_last_bva.csv': 'df_reg_demo_last_bva.csv',
                'df_reg_demo_last_lva.csv': 'df_reg_demo_last_lva.csv',
                
                # Session processor outputs (in data/output/output/)
                'session_this_filtered_valid_cols.csv': 'session_this_filtered_valid_cols.csv',
                'session_last_filtered_valid_cols_bva.csv': 'session_last_filtered_valid_cols_bva.csv',
                'session_last_filtered_valid_cols_lva.csv': 'session_last_filtered_valid_cols_lva.csv',
                
                # Scan processor outputs
                'scan_this_filtered_valid_cols.csv': 'scan_this_filtered_valid_cols.csv',
                'scan_bva_past.csv': 'scan_bva_past.csv',
                'scan_lva_past.csv': 'scan_lva_past.csv',
                'sessions_visited_last_bva.csv': 'sessions_visited_last_bva.csv',
                'sessions_visited_last_lva.csv': 'sessions_visited_last_lva.csv',
                
                # Streams file from session processor
                'streams.csv': 'streams.csv',
                'streams.json': 'streams.json'
            }
            
            # Try to find and copy files from any of the input paths
            for input_key in ['input_registration', 'input_scan', 'input_session']:
                if input_key in input_paths and input_paths[input_key]:
                    input_path = input_paths[input_key]
                    
                    if os.path.exists(input_path) and os.path.isdir(input_path):
                        # Look for expected files
                        for filename in os.listdir(input_path):
                            src_file = os.path.join(input_path, filename)
                            
                            if os.path.isfile(src_file):
                                # Copy to the event output directory
                                dst_file = os.path.join(output_dir, filename)
                                shutil.copy2(src_file, dst_file)
                                copied_files.append(filename)
                                self.logger.info(f"Copied {filename} to {output_dir}")
                                
                                # Also copy to data/output for compatibility
                                alt_dst = os.path.join(alt_output_dir, filename)
                                shutil.copy2(src_file, alt_dst)
            
            # CRITICAL: Change working directory to azureml_pipeline
            # This is necessary because the PA processors use relative paths
            os.chdir(root_dir)
            self.logger.info(f"Changed working directory to: {root_dir}")
            
            # Now check if the expected files exist in the right locations
            missing_files = []
            for expected_file in expected_files.keys():
                file_path = os.path.join(output_dir, expected_file)
                if not os.path.exists(file_path):
                    # Try alternate names and locations
                    alt_path = os.path.join(alt_output_dir, expected_file)
                    if os.path.exists(alt_path):
                        # Copy to expected location
                        shutil.copy2(alt_path, file_path)
                        self.logger.info(f"Copied {expected_file} from alternate location")
                        copied_files.append(expected_file)
                    else:
                        missing_files.append(expected_file)
                        self.logger.warning(f"Missing expected file: {expected_file}")
            
            if missing_files:
                self.logger.warning(f"Missing {len(missing_files)} expected files: {missing_files}")
                self.logger.warning("Neo4j processors may fail due to missing input files")
            
            if copied_files:
                self.logger.info(f"Successfully prepared {len(copied_files)} files for Neo4j processing")
                return True
            else:
                self.logger.warning("No files were copied from Step 1 outputs")
                self.logger.warning("This indicates Step 1 may not have completed successfully")
                return False
                
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
    
    try:
        # Initialize the step
        step = Neo4jPreparationStep(args.config, args.incremental)
        
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
        action='store_true',
        help='Run in incremental mode (only create new nodes/relationships)'
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