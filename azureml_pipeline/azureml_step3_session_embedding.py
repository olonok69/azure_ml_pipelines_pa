#!/usr/bin/env python
"""
Azure ML Pipeline Step 3: Session Embedding
Generates and stores text embeddings for session nodes in Neo4j.
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

# Import PA processor
from PA.session_embedding_processor import SessionEmbeddingProcessor
from PA.utils.config_utils import load_config
from PA.utils.logging_utils import setup_logging
from PA.utils.keyvault_utils import ensure_env_file, KeyVaultManager
from PA.utils.app_insights import configure_app_insights
from neo4j_env_utils import apply_neo4j_credentials, select_neo4j_environment
from step_input_sync import stage_step1_outputs


class SessionEmbeddingStep:
    """Azure ML Session Embedding Step for Personal Agendas pipeline."""
    
    def __init__(
        self,
        config_path: str,
        incremental: bool = False,
        use_keyvault: bool = True,
        neo4j_environment: Optional[str] = None,
    ):
        """
        Initialize the Session Embedding Step.
        
        Args:
            config_path: Path to configuration file
            incremental: Whether to run incremental processing (create_only_new)
            use_keyvault: Whether to use Azure Key Vault for secrets
        """
        self.config_path = config_path
        self.incremental = incremental
        self.use_keyvault = use_keyvault
        cleaned_override = (neo4j_environment or "").strip()
        self.neo4j_environment_override = cleaned_override or None
        self.logger = self._setup_logging()
        
        self.config = self._load_configuration(config_path)
        self.selected_neo4j_environment = self._apply_environment_override()
        # For embedding processor, incremental means create_only_new
        self.create_only_new = self.incremental

        # Load secrets from Key Vault if in Azure ML
        if self.use_keyvault and self._is_azure_ml_environment():
            self._load_secrets_from_keyvault()
    
    def _is_azure_ml_environment(self) -> bool:
        """Check if running in Azure ML environment."""
        return os.environ.get('AZUREML_RUN_ID') is not None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_secrets_from_keyvault(self):
        """Load secrets from Azure Key Vault."""
        try:
            self.logger.info("Loading secrets from Azure Key Vault")
            existing_uri = os.environ.get("NEO4J_URI")
            existing_user = os.environ.get("NEO4J_USERNAME")
            existing_pwd = os.environ.get("NEO4J_PASSWORD")

            if all([existing_uri, existing_user, existing_pwd]):
                apply_neo4j_credentials(
                    self.config,
                    existing_uri,
                    existing_user,
                    existing_pwd,
                    logger=self.logger,
                    environment_override=self.selected_neo4j_environment,
                )
                self.logger.info("Neo4j credentials already available in environment; skipping Key Vault")
                return
            
            # Get Key Vault name from environment or use default
            kv_name = os.environ.get("KEYVAULT_NAME", "strategicai-kv-uks-dev")
            
            # Create Key Vault Manager
            kv_manager = KeyVaultManager(kv_name)
            
            # Get Neo4j credentials
            neo4j_secrets = {
                "NEO4J_URI": kv_manager.get_secret("neo4j-uri"),
                "NEO4J_USERNAME": kv_manager.get_secret("neo4j-username"),
                "NEO4J_PASSWORD": kv_manager.get_secret("neo4j-password")
            }
            
            # Set as environment variables
            for key, value in neo4j_secrets.items():
                if value:
                    os.environ[key] = value
                    self.logger.info(f"Loaded {key} from Key Vault")
            
            apply_neo4j_credentials(
                self.config,
                neo4j_secrets.get("NEO4J_URI"),
                neo4j_secrets.get("NEO4J_USERNAME"),
                neo4j_secrets.get("NEO4J_PASSWORD"),
                logger=self.logger,
                environment_override=self.selected_neo4j_environment,
            )

            self.logger.info("Successfully loaded all secrets from Key Vault")

        except Exception as e:
            self.logger.warning(f"Could not load secrets from Key Vault: {e}")
            self.logger.info("Will fall back to environment variables or .env file")
    
    def _apply_environment_override(self) -> Optional[str]:
        """Ensure the config uses the highest-priority Neo4j environment selection."""
        config_env = (self.config.get('neo4j', {}) or {}).get('environment')
        env_lower, _ = select_neo4j_environment(
            self.neo4j_environment_override, config_env, logger=self.logger
        )

        if env_lower:
            self.config.setdefault('neo4j', {})['environment'] = env_lower
            self.logger.info("Using Neo4j environment '%s'", env_lower)
            return env_lower

        if self.neo4j_environment_override:
            self.logger.warning(
                "Ignoring invalid Neo4j environment override '%s'",
                self.neo4j_environment_override,
            )

        self.config.setdefault('neo4j', {}).pop('environment', None)
        self.logger.info("Neo4j environment not specified; defaulting to fallback credentials")
        return None

    def _load_configuration(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        self.logger.info(f"Loading configuration from: {config_path}")
        
        # Check if config path exists
        if not os.path.exists(config_path):
            self.logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration
        config = load_config(config_path)
        
        # Log configuration summary
        self.logger.info(f"Configuration loaded successfully")
        self.logger.info(f"Show name: {config.get('neo4j', {}).get('show_name', 'unknown')}")
        self.logger.info(f"Embedding model: {config.get('embeddings', {}).get('model', 'all-MiniLM-L6-v2')}")
        self.logger.info(f"Include stream descriptions: {config.get('embeddings', {}).get('include_stream_descriptions', False)}")
        
        return config
    
    def setup_data_directories(self, root_dir: str) -> Dict[str, str]:
        """
        Setup necessary data directories.
        
        Args:
            root_dir: Root directory for data
            
        Returns:
            Dictionary of directory paths
        """
        directories = {
            'data': os.path.join(root_dir, 'data'),
            'output': os.path.join(root_dir, 'data', 'output'),
            'logs': os.path.join(root_dir, 'logs')
        }
        
        for name, path in directories.items():
            os.makedirs(path, exist_ok=True)
            self.logger.info(f"Created/verified directory: {path}")
        
        return directories
    
    def run_session_embedding_processing(self) -> Dict[str, Any]:
        """Run session embedding processing."""
        try:
            self.logger.info("Initializing session embedding processor")
            processor = SessionEmbeddingProcessor(self.config)
            
            # Process embeddings
            result = processor.process(create_only_new=self.create_only_new)
            
            # Prepare result dictionary
            if result:
                result_dict = {
                    'status': 'success',
                    'statistics': processor.statistics if hasattr(processor, 'statistics') else {}
                }
            else:
                result_dict = {
                    'status': 'failed',
                    'error': 'Processing returned False'
                }
            
            self.logger.info(f"Session embedding processing completed: {result_dict.get('statistics', {})}")
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Error in session embedding processing: {e}")
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def save_outputs(self, results: Dict[str, Any], output_dir: str):
        """
        Save processing outputs to Azure ML output directory.
        
        Args:
            results: Results dictionary from processing
            output_dir: Azure ML output directory
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save results summary
            summary_path = os.path.join(output_dir, 'embedding_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Saved embedding summary to: {summary_path}")
            
            # Save statistics if available
            for processor_name, result in results.items():
                if isinstance(result, dict) and 'statistics' in result:
                    stats_path = os.path.join(output_dir, f'{processor_name}_statistics.json')
                    with open(stats_path, 'w') as f:
                        json.dump(result['statistics'], f, indent=2, default=str)
                    self.logger.info(f"Saved statistics to: {stats_path}")
            
            # Create a completion marker file
            marker_path = os.path.join(output_dir, 'embedding_complete.txt')
            with open(marker_path, 'w') as f:
                f.write(f"Session embedding processing completed at {datetime.now().isoformat()}\n")
                f.write(f"Status: {results.get('session_embedding', {}).get('status', 'unknown')}\n")
            
            self.logger.info(f"Created completion marker: {marker_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving outputs: {e}")
            traceback.print_exc()
    
    def process(self) -> Dict[str, Any]:
        """
        Main processing method for session embedding step.
        
        Returns:
            Dictionary containing results from the processor
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Session Embedding Step")
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Incremental: {self.incremental}")
        self.logger.info("=" * 60)
        
        # Setup directories
        directories = self.setup_data_directories(root_dir)
        
        # Initialize results
        results = {}
        
        # Run session embedding processing
        processors_config = self.config.get('processors', {})
        
        if processors_config.get('session_embedding_processing', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 40)
            self.logger.info("SESSION EMBEDDING PROCESSING")
            self.logger.info("=" * 40)
            results['session_embedding'] = self.run_session_embedding_processing()
        else:
            self.logger.info("Session embedding processing is disabled in configuration")
            results['session_embedding'] = {
                'status': 'skipped',
                'reason': 'Disabled in configuration'
            }
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Session Embedding Step Completed")
        self.logger.info("=" * 60)
        
        return results


def main(args):
    """Main entry point for Azure ML step."""
    print("\n" + "=" * 60)
    print("AZURE ML SESSION EMBEDDING STEP")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    configure_app_insights(service_name="pa_step3_session_embedding")
    
    try:
        # Initialize the step
        step = SessionEmbeddingStep(
            args.config,
            incremental=args.incremental,
            neo4j_environment=args.neo4j_environment,
        )

        input_paths = {
            'input_registration': args.input_registration,
            'input_scan': args.input_scan,
            'input_session': args.input_session,
        }

        if any(input_paths.values()):
            data_dir = os.path.join(root_dir, 'data')
            stage_step1_outputs(step.config, input_paths, data_dir, step.logger)
        
        # Run session embedding step
        results = step.process()
        
        # Save outputs to Azure ML location
        step.save_outputs(results, args.output_metadata)
        
        print("\n" + "=" * 60)
        print("SESSION EMBEDDING STEP SUMMARY")
        print("=" * 60)
        print(f"Configuration: {args.config}")
        print(f"Incremental: {args.incremental}")
        resolved_env = step.selected_neo4j_environment or "default"
        print(f"Neo4j environment (effective): {resolved_env}")
        print(f"Results:")
        
        # Print summary
        embedding_result = results.get('session_embedding', {})
        status = embedding_result.get('status', 'unknown')
        print(f"  - Status: {status}")
        
        # Print statistics if available
        stats = embedding_result.get('statistics', {})
        if stats:
            print(f"    Total sessions processed: {stats.get('total_sessions_processed', 0)}")
            print(f"    Sessions with embeddings: {stats.get('sessions_with_embeddings', 0)}")
            print(f"    Sessions with stream descriptions: {stats.get('sessions_with_stream_descriptions', 0)}")
            
            session_types = stats.get('sessions_by_type', {})
            if session_types:
                print(f"    Sessions this year: {session_types.get('sessions_this_year', 0)}")
                print(f"    Sessions past year: {session_types.get('sessions_past_year', 0)}")
        
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
    parser = argparse.ArgumentParser(description="Azure ML Session Embedding Step")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--incremental",
        nargs="?",
        const=True,
        default=False,
        type=_bool_arg,
        help="Run incremental processing (accepts true/false)"
    )

    parser.add_argument(
        "--neo4j_environment",
        type=str,
        help="Override Neo4j environment (dev|test|prod)"
    )
    
    parser.add_argument(
        "--output_metadata",
        type=str,
        required=True,
        help="Output directory for metadata and results"
    )

    parser.add_argument(
        "--input_registration",
        type=str,
        help="Path to Step 1 registration outputs"
    )

    parser.add_argument(
        "--input_scan",
        type=str,
        help="Path to Step 1 scan outputs"
    )

    parser.add_argument(
        "--input_session",
        type=str,
        help="Path to Step 1 session outputs"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)