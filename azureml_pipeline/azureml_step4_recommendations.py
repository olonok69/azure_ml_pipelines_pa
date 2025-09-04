#!/usr/bin/env python
"""
Azure ML Pipeline Step 4: Recommendations
Generates session recommendations for visitors based on embeddings and similarity.
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

import pandas as pd
from dotenv import load_dotenv

# Azure ML imports
from azure.identity import DefaultAzureCredential
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
        
        # Load secrets from Key Vault if in Azure ML
        if self.use_keyvault and self._is_azure_ml_environment():
            self._load_secrets_from_keyvault()
        
        self.config = self._load_configuration(config_path)
        # For recommendation processor, incremental means create_only_new
        self.create_only_new = self.incremental
    
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
            keyvault_name = os.environ.get('KEYVAULT_NAME', 'strategicai-kv-uks-dev')
            self.logger.info(f"Loading secrets from Key Vault: {keyvault_name}")
            
            kv_manager = KeyVaultManager(keyvault_name)
            
            # Load Neo4j secrets
            neo4j_secrets = {
                'NEO4J_URI': 'neo4j-uri',
                'NEO4J_USERNAME': 'neo4j-username', 
                'NEO4J_PASSWORD': 'neo4j-password'
            }
            
            for env_var, secret_name in neo4j_secrets.items():
                if not os.environ.get(env_var):
                    secret_value = kv_manager.get_secret(secret_name)
                    if secret_value:
                        os.environ[env_var] = secret_value
                        self.logger.info(f"Loaded {env_var} from Key Vault")
            
            # Load OpenAI API key if using LangChain
            if not os.environ.get('OPENAI_API_KEY'):
                openai_key = kv_manager.get_secret('openai-api-key')
                if openai_key:
                    os.environ['OPENAI_API_KEY'] = openai_key
                    self.logger.info("Loaded OpenAI API key from Key Vault")
                    
        except Exception as e:
            self.logger.warning(f"Could not load secrets from Key Vault: {e}")
            self.logger.info("Will use environment variables or config file")
    
    def _load_configuration(self, config_path: str) -> Dict:
        """Load configuration file."""
        try:
            self.logger.info(f"Loading configuration from: {config_path}")
            config = load_config(config_path)
            
            # Update Neo4j connection from environment if available
            if os.environ.get('NEO4J_URI'):
                config['neo4j']['uri'] = os.environ['NEO4J_URI']
            if os.environ.get('NEO4J_USERNAME'):
                config['neo4j']['username'] = os.environ['NEO4J_USERNAME']
            if os.environ.get('NEO4J_PASSWORD'):
                config['neo4j']['password'] = os.environ['NEO4J_PASSWORD']
            
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def setup_data_directories(self, root_dir: str) -> Dict[str, str]:
        """
        Setup required data directories.
        
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
    
    def run_recommendation_processing(self) -> Dict[str, Any]:
        """Run session recommendation processing."""
        try:
            self.logger.info("Initializing session recommendation processor")
            processor = SessionRecommendationProcessor(self.config)
            
            # Ensure CSV output is enabled in config
            if 'recommendation' not in self.config:
                self.config['recommendation'] = {}
            self.config['recommendation']['save_csv'] = True
            
            # Process recommendations
            result = processor.process(create_only_new=self.create_only_new)
            
            # Prepare result dictionary
            result_dict = {
                'status': 'success' if result else 'failed',
                'statistics': processor.statistics if hasattr(processor, 'statistics') else {},
                'output_file': None,
                'output_files': {}
            }
            
            # Check for output files
            event_name = self.config.get('event', {}).get('event_name', 'unknown')
            show_name = self.config.get('neo4j', {}).get('show_name', 'unknown')
            output_dir = Path(self.config.get('output_dir', 'data/output'))
            
            # Look for JSON and CSV files
            json_pattern = output_dir / f"recommendations/visitor_recommendations_{show_name}_*.json"
            json_files = list(output_dir.glob(f"recommendations/visitor_recommendations_*.json"))
            
            if json_files:
                # Get the most recent file
                most_recent_json = max(json_files, key=lambda p: p.stat().st_mtime)
                result_dict['output_file'] = str(most_recent_json)
                result_dict['output_files']['json'] = str(most_recent_json)
                
                # Check for corresponding CSV
                csv_file = Path(str(most_recent_json).replace('.json', '.csv'))
                if csv_file.exists():
                    result_dict['output_files']['csv'] = str(csv_file)
                    self.logger.info(f"Found CSV recommendations: {csv_file}")
                
                # Check for corresponding Parquet
                parquet_file = Path(str(most_recent_json).replace('.json', '.parquet'))
                if parquet_file.exists():
                    result_dict['output_files']['parquet'] = str(parquet_file)
                    self.logger.info(f"Found Parquet recommendations: {parquet_file}")
                
                self.logger.info(f"JSON recommendations saved to: {most_recent_json}")
                
                # Load and get basic stats
                try:
                    with open(most_recent_json, 'r') as f:
                        data = json.load(f)
                        if 'recommendations' in data:
                            result_dict['statistics']['total_rows'] = len(data['recommendations'])
                            result_dict['statistics']['unique_visitors'] = len(data['recommendations'])
                except:
                    pass
            
            self.logger.info(f"Recommendation processing completed with statistics: {result_dict.get('statistics', {})}")
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Error in recommendation processing: {e}")
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
            
            # Get event name for file naming
            event_name = self.config.get('event', {}).get('event_name', 'unknown')
            show_name = self.config.get('neo4j', {}).get('show_name', 'unknown')
            
            # Look for the actual recommendations files in the local output directory
            output_path = Path(self.config.get('output_dir', 'data/output'))
            recommendations_dir = output_path / 'recommendations'
            
            if recommendations_dir.exists():
                # Find all recommendation files (json, csv, parquet)
                json_files = list(recommendations_dir.glob(f"visitor_recommendations_{show_name}_*.json"))
                
                if json_files:
                    # Get the most recent JSON file
                    most_recent_json = max(json_files, key=lambda p: p.stat().st_mtime)
                    base_name = most_recent_json.stem  # Get filename without extension
                    
                    # Copy JSON file
                    dest_json = os.path.join(output_dir, f"{base_name}.json")
                    shutil.copy2(most_recent_json, dest_json)
                    self.logger.info(f"Copied JSON recommendations to: {dest_json}")
                    
                    # Look for and copy CSV file
                    csv_file = recommendations_dir / f"{base_name}.csv"
                    if csv_file.exists():
                        dest_csv = os.path.join(output_dir, f"{base_name}.csv")
                        shutil.copy2(csv_file, dest_csv)
                        self.logger.info(f"Copied CSV recommendations to: {dest_csv}")
                    else:
                        self.logger.warning(f"CSV file not found: {csv_file}")
                    
                    # Look for and copy Parquet file
                    parquet_file = recommendations_dir / f"{base_name}.parquet"
                    if parquet_file.exists():
                        dest_parquet = os.path.join(output_dir, f"{base_name}.parquet")
                        shutil.copy2(parquet_file, dest_parquet)
                        self.logger.info(f"Copied Parquet recommendations to: {dest_parquet}")
                    
                    # Update results to indicate files were copied
                    if 'recommendations' not in results:
                        results['recommendations'] = {}
                    results['recommendations']['output_files'] = {
                        'json': f"{base_name}.json",
                        'csv': f"{base_name}.csv" if csv_file.exists() else None,
                        'parquet': f"{base_name}.parquet" if parquet_file.exists() else None
                    }
            
            # Save results summary with updated file info
            summary_path = os.path.join(output_dir, 'recommendations_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Saved recommendations summary to: {summary_path}")
            
            # Save statistics if available
            for processor_name, result in results.items():
                if isinstance(result, dict) and 'statistics' in result:
                    stats_path = os.path.join(output_dir, f'{processor_name}_statistics.json')
                    with open(stats_path, 'w') as f:
                        json.dump(result['statistics'], f, indent=2, default=str)
                    self.logger.info(f"Saved statistics to: {stats_path}")
            
            # Create a detailed completion marker file
            marker_path = os.path.join(output_dir, 'recommendations_complete.txt')
            with open(marker_path, 'w') as f:
                f.write(f"Recommendations processing completed at {datetime.now().isoformat()}\n")
                f.write(f"Status: {results.get('recommendations', {}).get('status', 'unknown')}\n")
                stats = results.get('recommendations', {}).get('statistics', {})
                f.write(f"Visitors processed: {stats.get('visitors_processed', 0)}\n")
                f.write(f"Visitors with recommendations: {stats.get('visitors_with_recommendations', 0)}\n")
                f.write(f"Total recommendations generated: {stats.get('total_recommendations_generated', 0)}\n")
                f.write("\nOutput Files:\n")
                
                # List all files in output directory
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path) / 1024  # Size in KB
                        f.write(f"  - {file} ({size:.2f} KB)\n")
            
            self.logger.info(f"Created completion marker: {marker_path}")
            
            # Upload to blob storage
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
        Upload recommendation files to blob storage in the same location as input data.
        
        Args:
            output_dir: Directory containing the files to upload
        """
        try:
            from azure.storage.blob import BlobServiceClient
            from azure.identity import DefaultAzureCredential
            
            # Get configuration
            event_name = self.config.get('event', {}).get('event_name', 'ecomm')
            
            # Parse the storage account from environment or use default
            storage_account = os.environ.get('STORAGE_ACCOUNT_NAME', 'strategicaistuksdev02')
            container_name = 'landing'
            
            # Build the blob storage URL
            account_url = f"https://{storage_account}.blob.core.windows.net"
            
            # Create blob service client using DefaultAzureCredential (works in Azure ML)
            credential = DefaultAzureCredential()
            blob_service_client = BlobServiceClient(account_url, credential=credential)
            container_client = blob_service_client.get_container_client(container_name)
            
            # Find recommendation files to upload
            files_to_upload = []
            for file in os.listdir(output_dir):
                if file.startswith('visitor_recommendations_') and file.endswith(('.json', '.csv', '.parquet')):
                    files_to_upload.append(file)
            
            if not files_to_upload:
                self.logger.warning("No recommendation files found to upload")
                return
            
            # Extract timestamp from filename (e.g., visitor_recommendations_ecomm_20250904_134436.json)
            # Use the first file to get the timestamp
            timestamp_part = files_to_upload[0].replace('visitor_recommendations_', '').replace(f'{event_name}_', '').split('.')[0]
            
            # Create the blob path: data/{event_name}/recommendations/visitor_recommendations_{event_name}_{timestamp}/
            blob_folder = f"data/{event_name}/recommendations/visitor_recommendations_{event_name}_{timestamp_part}"
            
            # Upload each file
            for filename in files_to_upload:
                file_path = os.path.join(output_dir, filename)
                blob_name = f"{blob_folder}/{filename}"
                
                try:
                    with open(file_path, 'rb') as data:
                        blob_client = container_client.get_blob_client(blob_name)
                        blob_client.upload_blob(data, overwrite=True)
                        self.logger.info(f"Uploaded to blob: {blob_name}")
                except Exception as e:
                    self.logger.error(f"Failed to upload {filename}: {e}")
            
            self.logger.info(f"Successfully uploaded {len(files_to_upload)} files to blob storage")
            self.logger.info(f"Blob folder: {blob_folder}")
            
            # Create a marker file with the blob location
            marker_path = os.path.join(output_dir, 'blob_upload_info.txt')
            with open(marker_path, 'w') as f:
                f.write(f"Files uploaded to blob storage at {datetime.now().isoformat()}\n")
                f.write(f"Storage Account: {storage_account}\n")
                f.write(f"Container: {container_name}\n")
                f.write(f"Folder: {blob_folder}\n")
                f.write("\nUploaded files:\n")
                for filename in files_to_upload:
                    f.write(f"  - {filename}\n")
            
        except ImportError:
            self.logger.warning("Azure Storage Blob library not available - skipping blob upload")
        except Exception as e:
            self.logger.error(f"Error uploading to blob storage: {e}")
            # Don't fail the whole step if blob upload fails
            self.logger.info("Continuing despite blob upload failure - files are still in Azure ML outputs")
    
    def process(self) -> Dict[str, Any]:
        """
        Main processing method for recommendations step.
        
        Returns:
            Dictionary containing results from the processor
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Recommendations Step")
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Incremental: {self.incremental}")
        self.logger.info("=" * 60)
        
        # Setup directories
        directories = self.setup_data_directories(root_dir)
        
        # Initialize results
        results = {}
        
        # Run recommendation processing
        processors_config = self.config.get('processors', {})
        
        if processors_config.get('session_recommendation_processing', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 40)
            self.logger.info("SESSION RECOMMENDATION PROCESSING")
            self.logger.info("=" * 40)
            results['recommendations'] = self.run_recommendation_processing()
        else:
            self.logger.info("Session recommendation processing is disabled in configuration")
            results['recommendations'] = {
                'status': 'skipped',
                'reason': 'Disabled in configuration'
            }
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Recommendations Step Completed")
        self.logger.info("=" * 60)
        
        return results


def main(args):
    """Main entry point for Azure ML step."""
    print("\n" + "=" * 60)
    print("AZURE ML RECOMMENDATIONS STEP")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize MLflow tracking (following PA/main.py pattern)
    mlflow_manager = None
    try:
        from PA.utils.mlflow_utils import MLflowManager
        
        # Check if MLflow environment variables are set
        mlflow_experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
        databricks_host = os.environ.get("DATABRICKS_HOST")
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        
        if mlflow_experiment_id and databricks_host and mlflow_tracking_uri:
            print("Initializing MLflow tracking")
            mlflow_manager = MLflowManager()
            
            # Verify MLflow environment
            if mlflow_manager.verify_environment():
                print("MLflow tracking enabled successfully")
                
                # Start MLflow run (we'll use a simplified version since we don't have full config yet)
                import mlflow
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                mlflow.set_experiment(experiment_id=mlflow_experiment_id)
                
                # Start a run for Step 4
                with mlflow.start_run(run_name=f"step4_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    # Log basic parameters
                    mlflow.log_param("step", "recommendations")
                    mlflow.log_param("config_path", args.config)
                    mlflow.log_param("incremental", args.incremental)
                    
                    # Initialize and run the step
                    step = RecommendationsStep(args.config, args.incremental)
                    results = step.process()
                    
                    # Log metrics from results
                    if mlflow_manager and results.get('recommendations'):
                        stats = results.get('recommendations', {}).get('statistics', {})
                        if stats:
                            # Log Step 10 metrics (following PA/main.py pattern)
                            step_metrics = {
                                'step10_visitors_processed': stats.get('visitors_processed', 0),
                                'step10_visitors_with_recommendations': stats.get('visitors_with_recommendations', 0),
                                'step10_visitors_without_recommendations': stats.get('visitors_without_recommendations', 0),
                                'step10_recommendations_generated': stats.get('total_recommendations_generated', 0),
                                'step10_filtered_recommendations': stats.get('total_filtered_recommendations', 0),
                                'step10_errors': stats.get('errors', 0),
                                'step10_processing_time': stats.get('processing_time', 0)
                            }
                            
                            for metric_name, metric_value in step_metrics.items():
                                if isinstance(metric_value, (int, float)):
                                    mlflow.log_metric(metric_name, metric_value)
                            
                            print(f"Logged {len(step_metrics)} metrics to MLflow")
                    
                    # Save outputs to Azure ML location
                    step.save_outputs(results, args.output_metadata)
                    
            else:
                raise Exception("MLflow environment verification failed")
        else:
            print("MLflow environment variables not set. Continuing without MLflow tracking.")
            print("To enable MLflow, set: MLFLOW_EXPERIMENT_ID, DATABRICKS_HOST, MLFLOW_TRACKING_URI")
            
            # Run without MLflow
            step = RecommendationsStep(args.config, args.incremental)
            results = step.process()
            step.save_outputs(results, args.output_metadata)
            
    except ImportError as e:
        print(f"MLflow utilities not available: {e}")
        print("Continuing without MLflow tracking")
        
        # Run without MLflow
        step = RecommendationsStep(args.config, args.incremental)
        results = step.process()
        step.save_outputs(results, args.output_metadata)
        
    except Exception as e:
        if "MLflow" in str(e):
            print(f"Failed to initialize MLflow: {e}")
            print("MLFLOW_EXPERIMENT_ID not set. Continuing without MLflow tracking.")
            
            # Run without MLflow
            step = RecommendationsStep(args.config, args.incremental)
            results = step.process()
            step.save_outputs(results, args.output_metadata)
        else:
            # Re-raise non-MLflow exceptions
            print(f"JOB FAILED: {str(e)}")
            traceback.print_exc()
            raise
    
    # Print summary (runs regardless of MLflow status)
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS STEP SUMMARY")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Incremental: {args.incremental}")
    print(f"MLflow Tracking: {'Enabled' if mlflow_manager else 'Disabled'}")
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
        action="store_true",
        help="Run incremental processing (only create new recommendations)"
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
    main(args)