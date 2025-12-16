#!/usr/bin/env python
"""
Azure ML Pipeline Step 4: Recommendations
Generates session recommendations for visitors based on embeddings and similarity.
FIXED: 
1. Improved authentication handling for blob storage upload
2. Made MLflow tracking optional with proper Azure ML integration
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
from typing import Dict, List, Any, Optional
import time

import pandas as pd
from dotenv import load_dotenv

# Azure ML imports
from azure.identity import (
    DefaultAzureCredential, 
    ManagedIdentityCredential, 
    EnvironmentCredential,
    ClientSecretCredential,
    ChainedTokenCredential
)
from azure.ai.ml import MLClient


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
from PA.session_recommendation_processor import SessionRecommendationProcessor
from PA.utils.config_utils import load_config
from PA.utils.logging_utils import setup_logging
from PA.utils.keyvault_utils import ensure_env_file, KeyVaultManager
import mlflow
from neo4j import GraphDatabase


def _in_azureml_run() -> bool:
    return bool(os.getenv("AZUREML_RUN_ID"))

def configure_mlflow(logger):
    if not _in_azureml_run():
        logger.info("Not inside Azure ML run context – leaving MLflow config unchanged")
        return
    if os.getenv("FORCE_AZUREML_MLFLOW", "false").lower() == "true":
        if os.getenv("MLFLOW_TRACKING_URI", "").startswith("databricks"):
            os.environ.pop("MLFLOW_TRACKING_URI", None)
            logger.info("Removed Databricks MLFLOW_TRACKING_URI for Azure ML logging")
    try:
        mlflow.set_experiment("personal_agendas_complete")
        # Ensure an active run (Azure ML usually auto-starts; guard anyway)
        if mlflow.active_run() is None:
            mlflow.start_run()
        logger.info("MLflow experiment & run ready")
    except Exception as e:
        logger.warning(f"MLflow configuration warning: {e}")

class RecommendationsStep:
    """Azure ML Recommendations Step for Personal Agendas pipeline."""
    
    def __init__(self, config_path: str, incremental: bool = False, use_keyvault: bool = True):
        """
        Initialize the Recommendations Step.
        
        Args:
            config_path: Path to configuration file
            incremental: Whether to run incremental processing (create_only_new)
            use_keyvault: Whether to use Azure Key Vault for secrets
        """
        self.config_path = config_path
        self.incremental = incremental
        self.use_keyvault = use_keyvault
        self.logger = self._setup_logging()

        # Load config early
        self.config = self._load_configuration(config_path)
        self.create_only_new = self.incremental

        # Robust secret load (non-fatal). Do not declare success unless something found.
        if self.use_keyvault and self._is_azure_ml_environment():
            self._load_keyvault_secrets()
    
    def _is_azure_ml_environment(self) -> bool:
        """Check if running in Azure ML environment."""
        return os.environ.get('AZUREML_RUN_ID') is not None
    
    def _load_keyvault_secrets(self):
        """Load secrets from Key Vault or environment variables.
        This method now uses the same approach as Step 1 to avoid authentication issues.
        """
        try:
            # First check if Neo4j credentials are already in environment
            neo4j_uri = os.environ.get("NEO4J_URI")
            neo4j_username = os.environ.get("NEO4J_USERNAME")
            neo4j_password = os.environ.get("NEO4J_PASSWORD")
            
            if all([neo4j_uri, neo4j_username, neo4j_password]):
                self.logger.info("Neo4j credentials found in environment variables")
                return {
                    "NEO4J_URI": neo4j_uri,
                    "NEO4J_USERNAME": neo4j_username,
                    "NEO4J_PASSWORD": "***"
                }
            
            # If not all credentials are in environment, try to load from Key Vault
            kv_name = os.getenv("KEYVAULT_NAME", "strategicai-kv-uks-dev")
            self.logger.info(f"Neo4j credentials not complete in environment, trying Key Vault: {kv_name}")
            
            # Use ensure_env_file like Step 1 does
            env_path = os.path.join(project_root, "PA", "keys", ".env")
            if ensure_env_file(kv_name, env_path):
                # Load the created .env file
                load_dotenv(env_path)
                self.logger.info("Successfully loaded secrets from Key Vault via .env file")
                
                # Check if credentials are now available
                neo4j_uri = os.environ.get("NEO4J_URI")
                neo4j_username = os.environ.get("NEO4J_USERNAME")
                neo4j_password = os.environ.get("NEO4J_PASSWORD")
                
                if all([neo4j_uri, neo4j_username, neo4j_password]):
                    return {
                        "NEO4J_URI": neo4j_uri,
                        "NEO4J_USERNAME": neo4j_username,
                        "NEO4J_PASSWORD": "***"
                    }
            else:
                self.logger.warning("Could not load secrets from Key Vault, relying on environment variables")
                
        except Exception as e:
            self.logger.error(f"Error loading secrets: {e}")
            self.logger.info("Falling back to environment variables")
        
        return {}
    
    def _ensure_neo4j_credentials(self):
        """Ensure Neo4j credentials are available from environment or Key Vault."""
        needed = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
        
        # Check if all credentials are already available
        missing = [k for k in needed if not os.getenv(k)]
        
        if missing:
            self.logger.info(f"Missing credentials: {missing}, attempting to load...")
            self._load_keyvault_secrets()
            
            # Check again after loading
            missing = [k for k in needed if not os.getenv(k)]
        
        # Clean up any whitespace
        for k in needed:
            if os.getenv(k):
                os.environ[k] = os.getenv(k).strip()
        
        if missing:
            raise RuntimeError(f"Missing Neo4j credentials after Key Vault/environment fallback: {missing}")

        uri = os.environ["NEO4J_URI"]
        user = os.environ["NEO4J_USERNAME"]
        pwd = os.environ["NEO4J_PASSWORD"]

        self.logger.info("Verifying Neo4j connectivity...")
        try:
            drv = GraphDatabase.driver(uri, auth=(user, pwd))
            drv.verify_connectivity()
            self.logger.info("Neo4j connectivity verified")
        except Exception as e:
            raise RuntimeError(f"Neo4j connectivity test failed: {e}") from e
        finally:
            try: drv.close()
            except: pass

        self.config.setdefault("neo4j", {})
        self.config["neo4j"].update({"uri": uri, "username": user, "password": pwd})
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _get_azure_credential(self):
        """
        Get the appropriate Azure credential based on the environment.
        Tries multiple authentication methods in order of preference.
        """
        credentials_to_try = []
        
        # 1. First try service principal if credentials are available
        client_id = os.environ.get("AZURE_CLIENT_ID")
        client_secret = os.environ.get("AZURE_CLIENT_SECRET")
        tenant_id = os.environ.get("AZURE_TENANT_ID")
        
        if all([client_id, client_secret, tenant_id]):
            self.logger.info("Using Service Principal credentials for authentication")
            sp_credential = ClientSecretCredential(
                client_id=client_id,
                client_secret=client_secret,
                tenant_id=tenant_id
            )
            credentials_to_try.append(sp_credential)
        
        # 2. Try Managed Identity (for Azure ML compute)
        if self._is_azure_ml_environment():
            self.logger.info("Adding Managed Identity to credential chain")
            mi_credential = ManagedIdentityCredential()
            credentials_to_try.append(mi_credential)
        
        # 3. Environment credential as fallback
        env_credential = EnvironmentCredential()
        credentials_to_try.append(env_credential)
        
        if credentials_to_try:
            # Use ChainedTokenCredential to try multiple auth methods
            return ChainedTokenCredential(*credentials_to_try)
        else:
            # Fall back to DefaultAzureCredential as last resort
            self.logger.warning("Using DefaultAzureCredential as fallback")
            return DefaultAzureCredential()
    
    
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
        self.logger.info(f"Event name: {config.get('event', {}).get('name', 'unknown')}")
        
        return config
    


    def setup_data_directories(self, root_dir: str) -> Dict[str, str]:
        """
        Setup necessary data directories.
        
        Args:
            root_dir: Root directory for the project
            
        Returns:
            Dictionary with directory paths
        """
        directories = {
            'data': os.path.join(root_dir, 'data'),
            'output': os.path.join(root_dir, 'data', 'output'),
            'recommendations': os.path.join(root_dir, 'data', 'output', 'recommendations'),
            'logs': os.path.join(root_dir, 'logs')
        }
        
        for dir_name, dir_path in directories.items():
            os.makedirs(dir_path, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")
        
        return directories
    
    def run_recommendation_processing(self) -> Dict[str, Any]:
        try:
            self.logger.info("Initializing Session Recommendation Processor")
            processor = SessionRecommendationProcessor(self.config)
            self.logger.info("Running recommendation processing...")
            processor.process()

            statistics = getattr(processor, 'statistics', {})
            result_dict = {'status': 'success', 'statistics': statistics, 'output_files': {}}

            show_name = self.config.get('event', {}).get('name', 'ecomm')
            output_dir = Path(root_dir) / 'data' / 'output' / 'recommendations'
            # Only match current show
            pattern = f"visitor_recommendations_{show_name}_*.json"
            json_files = list(output_dir.glob(pattern))

            if json_files:
                most_recent_json = max(json_files, key=lambda p: p.stat().st_mtime)
                result_dict['output_files']['json'] = str(most_recent_json)
                # Optional companion formats
                for ext in ('.csv', '.parquet'):
                    candidate = most_recent_json.with_suffix(ext)
                    if candidate.exists():
                        result_dict['output_files'][ext.lstrip('.')] = str(candidate)

                # Basic stats enrichment
                try:
                    with open(most_recent_json, 'r') as f:
                        data = json.load(f)
                        recs = data.get('recommendations', [])
                        result_dict['statistics'].setdefault('total_rows', len(recs))
                        # If structure is list of visitor objects, adjust as needed
                        result_dict['statistics'].setdefault('unique_visitors', len(recs))
                except Exception as e:
                    self.logger.debug(f"Could not enrich statistics from JSON: {e}")

            self.logger.info(f"Recommendation processing completed: {result_dict['statistics']}")
            return result_dict
        except Exception as e:
            self.logger.error(f"Error in recommendation processing: {e}")
            self.logger.debug(traceback.format_exc())
            return {'status': 'failed', 'error': str(e), 'statistics': {}}
    
    def save_outputs(self, results: Dict[str, Any], output_dir: str):
        """
        Save processing outputs to Azure ML output directory.
        
        Args:
            results: Processing results dictionary
            output_dir: Output directory path
        """
        try:
            self.logger.info(f"Saving outputs to: {output_dir}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Copy recommendation files to output directory
            if 'recommendations' in results and results['recommendations'].get('status') == 'success':
                output_files = results['recommendations'].get('output_files', {})
                
                for file_type, file_path in output_files.items():
                    if file_path and os.path.exists(file_path):
                        dest_path = os.path.join(output_dir, os.path.basename(file_path))
                        shutil.copy2(file_path, dest_path)
                        self.logger.info(f"Copied {file_type} recommendations to: {dest_path}")
            
            # Save results summary
            summary_path = os.path.join(output_dir, 'recommendations_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Saved recommendations summary to: {summary_path}")
            
            # Save statistics separately
            if 'recommendations' in results:
                stats = results['recommendations'].get('statistics', {})
                if stats:
                    stats_path = os.path.join(output_dir, 'recommendations_statistics.json')
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f, indent=2)
                    self.logger.info(f"Saved statistics to: {stats_path}")
            
            # Create a completion marker file
            marker_path = os.path.join(output_dir, 'recommendations_complete.txt')
            with open(marker_path, 'w') as f:
                f.write(f"Recommendations processing completed at {datetime.now().isoformat()}\n")
                f.write(f"Status: {results.get('recommendations', {}).get('status', 'unknown')}\n")
                f.write("\nOutput Files:\n")
                
                # List all files in output directory
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path) / 1024  # Size in KB
                        f.write(f"  - {file} ({size:.2f} KB)\n")
            
            self.logger.info(f"Created completion marker: {marker_path}")
            
            # Upload to datastore (same location as input files)
            self.upload_to_blob_storage(output_dir)
            
            # Log summary of what was saved
            self.logger.info("\n" + "="*50)
            self.logger.info("STEP 4 OUTPUT SUMMARY")
            self.logger.info("="*50)
            self.logger.info(f"Output directory: {output_dir}")
            self.logger.info("Files saved:")
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path) / 1024
                    self.logger.info(f"  ✓ {file} ({size:.2f} KB)")
            self.logger.info("="*50)
            
        except Exception as e:
            self.logger.error(f"Error saving outputs: {e}")
            traceback.print_exc()
    
    def upload_to_blob_storage(self, output_dir: str):
        """
        Upload recommendation files to blob storage with improved authentication handling.
        
        Args:
            output_dir: Directory containing the files to upload
        """
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                from azure.storage.blob import BlobServiceClient
                
                # Get configuration
                event_name = self.config.get('event', {}).get('name', 'ecomm')
                
                # Parse the storage account from environment or use default
                storage_account = os.environ.get('STORAGE_ACCOUNT_NAME', 'strategicaistuksdev02')
                
                # Try multiple container names in order of preference
                container_candidates = [
                    'azureml-blobstore-f4c6fe07-7138-457d-98db-3ba5d18852bf',  # Default AML datastore
                    'azureml',  # Common Azure ML container
                    'outputs',  # Alternative outputs container
                    'landing'   # Original container (might not exist)
                ]
                
                # Build the blob storage URL
                account_url = f"https://{storage_account}.blob.core.windows.net"
                
                # Get appropriate credential
                credential = self._get_azure_credential()
                
                # Create blob service client with the credential
                self.logger.info(f"Attempting blob storage connection (attempt {attempt + 1}/{max_retries})")
                blob_service_client = BlobServiceClient(account_url, credential=credential)
                
                # Find the first available container
                container_name = None
                container_client = None
                
                for candidate_container in container_candidates:
                    try:
                        test_container = blob_service_client.get_container_client(candidate_container)
                        # Try to get container properties to test authentication
                        container_props = test_container.get_container_properties()
                        container_name = candidate_container
                        container_client = test_container
                        self.logger.info(f"Successfully connected to container: {container_name}")
                        break
                    except Exception as e:
                        self.logger.debug(f"Container {candidate_container} not accessible: {e}")
                        continue
                
                if not container_client:
                    # If no predefined containers work, try to list containers and use the first one
                    try:
                        containers = blob_service_client.list_containers()
                        first_container = next(containers, None)
                        if first_container:
                            container_name = first_container['name']
                            container_client = blob_service_client.get_container_client(container_name)
                            self.logger.info(f"Using first available container: {container_name}")
                        else:
                            raise Exception("No accessible containers found in storage account")
                    except Exception as e:
                        raise Exception(f"Could not find any accessible container: {e}")
                
                # Find recommendation files to upload
                files_to_upload = []
                for file in os.listdir(output_dir):
                    if file.startswith('visitor_recommendations_') and file.endswith(('.json', '.csv', '.parquet')):
                        files_to_upload.append(file)
                
                if not files_to_upload:
                    self.logger.warning("No recommendation files found to upload")
                    return
                
                # Extract timestamp from filename
                timestamp_part = files_to_upload[0].replace('visitor_recommendations_', '').replace(f'{event_name}_', '').split('.')[0]
                
                # Create the blob path
                blob_folder = f"data/{event_name}/recommendations/visitor_recommendations_{event_name}_{timestamp_part}"
                
                # Upload each file
                successful_uploads = 0
                for filename in files_to_upload:
                    file_path = os.path.join(output_dir, filename)
                    blob_name = f"{blob_folder}/{filename}"
                    
                    try:
                        with open(file_path, 'rb') as data:
                            blob_client = container_client.get_blob_client(blob_name)
                            blob_client.upload_blob(data, overwrite=True)
                            self.logger.info(f"Uploaded to blob: {blob_name}")
                            successful_uploads += 1
                    except Exception as e:
                        self.logger.error(f"Failed to upload {filename}: {e}")
                
                self.logger.info(f"Successfully uploaded {successful_uploads}/{len(files_to_upload)} files to blob storage")
                self.logger.info(f"Container: {container_name}")
                self.logger.info(f"Blob folder: {blob_folder}")
                
                # Create a marker file with the blob location
                marker_path = os.path.join(output_dir, 'blob_upload_info.txt')
                with open(marker_path, 'w') as f:
                    f.write(f"Files uploaded to blob storage at {datetime.now().isoformat()}\n")
                    f.write(f"Storage Account: {storage_account}\n")
                    f.write(f"Container: {container_name}\n")
                    f.write(f"Folder: {blob_folder}\n")
                    f.write(f"Successful uploads: {successful_uploads}/{len(files_to_upload)}\n")
                    f.write("\nUploaded files:\n")
                    for filename in files_to_upload:
                        f.write(f"  - {filename}\n")
                
                # If we got here, upload was successful
                break
                
            except ImportError:
                self.logger.warning("Azure Storage Blob library not available - skipping blob upload")
                break
            except Exception as e:
                self.logger.error(f"Error uploading to blob storage (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    # Don't fail the whole step if blob upload fails after all retries
                    self.logger.warning("All blob upload attempts failed - files are still available in Azure ML outputs")
                    self.logger.info("Files are saved in the Azure ML output directory and can be accessed through the Azure ML Studio")
    
    def process(self) -> Dict[str, Any]:
        configure_mlflow(self.logger)
        self._ensure_neo4j_credentials()

        self.logger.info("=" * 60)
        self.logger.info("Starting Recommendations Step")
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Incremental: {self.incremental}")
        self.logger.info("=" * 60)

        self.setup_data_directories(root_dir)
        results = {}
        processors_cfg = self.config.get('processors', {})
        enabled = processors_cfg.get('session_recommendation_processing', {}).get('enabled', True)

        if enabled:
            self.logger.info("\n" + "=" * 40)
            self.logger.info("SESSION RECOMMENDATION PROCESSING")
            self.logger.info("=" * 40)
            results['recommendations'] = self.run_recommendation_processing()
        else:
            msg = "Session recommendation processing disabled in configuration"
            self.logger.info(msg)
            results['recommendations'] = {'status': 'skipped', 'reason': msg, 'statistics': {}}

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Recommendations Step Completed")
        self.logger.info("=" * 60)
        return results


def setup_azure_ml_mlflow():
    """
    Setup MLflow for Azure ML environment.
    Returns MLflow manager or None if not available.
    """
    try:
        import mlflow
        from azureml.core import Run
        
        # Check if we're in Azure ML
        try:
            run = Run.get_context()
            if hasattr(run, '_run_id') and run._run_id != 'OfflineRun':
                # We're in Azure ML - MLflow is automatically configured
                print("Detected Azure ML environment - MLflow is automatically configured")
                return True
        except:
            pass
        
        # Try custom MLflow configuration if environment variables are set
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            print(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
            return True
        
        return None
        
    except ImportError:
        print("MLflow not available")
        return None
    except Exception as e:
        print(f"Error setting up MLflow: {e}")
        return None


def log_metrics_to_mlflow(metrics: Dict[str, Any], prefix: str = "step4_"):
    """
    Log metrics to MLflow (either Azure ML or custom).
    
    Args:
        metrics: Dictionary of metrics to log
        prefix: Prefix for metric names
    """
    try:
        import mlflow
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_name = f"{prefix}{key}"
                mlflow.log_metric(metric_name, value)
                print(f"Logged metric: {metric_name} = {value}")
    except Exception as e:
        print(f"Could not log metrics to MLflow: {e}")


def main(args):
    """Main entry point for Azure ML step."""
    print("\n" + "=" * 60)
    print("AZURE ML RECOMMENDATIONS STEP")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize step
    step = RecommendationsStep(args.config, args.incremental)
    
    # Run processing
    try:
        results = step.process()
    except Exception as e:
        # Fatal pre-processing failure (likely credentials)
        print(f"FATAL: {e}")
        traceback.print_exc()
        results = {'recommendations': {'status': 'failed', 'error': str(e), 'statistics': {}}}

    step.save_outputs(results, args.output_metadata)

    # Remove duplicate MLflow setup (configure_mlflow already called)
    if mlflow.active_run():
        rec_stats = results.get('recommendations', {}).get('statistics', {})
        for k, v in rec_stats.items():
            if isinstance(v, (int, float)):
                try:
                    mlflow.log_metric(f"step4_{k}", v)
                except Exception:
                    pass
    
    # Print summary
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS STEP SUMMARY")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Incremental: {args.incremental}")

    print(f"Results:")
    
    # Print statistics
    rec_result = results.get('recommendations', {})
    status = rec_result.get('status', 'unknown')
    print(f"  - Status: {status}")
    
    stats = rec_result.get('statistics', {})
    if stats:
        print(f"  Statistics:")
        print(f"    Visitors processed: {stats.get('visitors_processed', 0)}")
        print(f"    Visitors with recommendations: {stats.get('visitors_with_recommendations', 0)}")
        print(f"    Visitors without recommendations: {stats.get('visitors_without_recommendations', 0)}")
        print(f"    Total recommendations generated: {stats.get('total_recommendations_generated', 0)}")
        print(f"    Total filtered recommendations: {stats.get('total_filtered_recommendations', 0)}")
        print(f"    Errors: {stats.get('errors', 0)}")
        
        if 'total_rows' in stats:
            print(f"    Output file total rows: {stats['total_rows']}")
        if 'unique_visitors' in stats:
            print(f"    Unique visitors in output: {stats['unique_visitors']}")
    
    print("=" * 60)
    
    # Return success
    return 0 if status == 'success' else 1


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
    parser = argparse.ArgumentParser(description="Azure ML Recommendations Step")
    
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
        "--output_metadata",
        type=str,
        required=True,
        help="Output directory for metadata and results"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))