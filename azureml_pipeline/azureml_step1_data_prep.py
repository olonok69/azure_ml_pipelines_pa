"""
Azure ML Pipeline Step 1: Data Preparation
Processes Registration, Scan, and Session data
"""

import os
import sys
import argparse
import logging
import mlflow
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azureml.fsspec import AzureMachineLearningFileSystem
from dotenv import load_dotenv

import os
import sys

# Fix import paths for Azure ML environment
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
pa_dir = os.path.join(parent_dir, 'PA')

# Add paths
sys.path.insert(0, parent_dir)
sys.path.insert(0, pa_dir)

# Import with fallback
try:
    from registration_processor import RegistrationProcessor
    from scan_processor import ScanProcessor
    from session_processor import SessionProcessor
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging
except ImportError:
    from PA.registration_processor import RegistrationProcessor
    from PA.scan_processor import ScanProcessor
    from PA.session_processor import SessionProcessor
    from PA.utils.config_utils import load_config
    from PA.utils.logging_utils import setup_logging


class DataPreparationStep:
    """
    Azure ML Pipeline Step 1: Data Preparation
    Handles registration, scan, and session processing
    """
    
    def __init__(self, config_path: str, incremental: bool = False):
        """
        Initialize the data preparation step.
        
        Args:
            config_path: Path to configuration file
            incremental: Whether to run incremental processing
        """
        self.logger = self._setup_logging()
        self.config = self._load_configuration(config_path)
        self.incremental = incremental
        self.output_artifacts: Dict[str, Any] = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration for Azure ML."""
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"azureml_data_prep_{timestamp_str}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_configuration(self, config_path: str) -> Dict:
        """
        Load and validate configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        self.logger.info(f"Loading configuration from {config_path}")
        
        # Load base configuration
        config = load_config(config_path)
        
        # Add Azure ML specific settings if not present
        if 'azure_ml' not in config:
            config['azure_ml'] = {
                'step_name': 'data_preparation',
                'outputs': {
                    'registration': 'registration_output',
                    'scan': 'scan_output', 
                    'session': 'session_output',
                    'metadata': 'metadata_output'
                }
            }
        
        return config
    
    def download_blob_data(self, uri: str, local_dir: str) -> List[str]:
        """
        Download data from Azure Blob Storage.
        
        Args:
            uri: Azure blob URI
            local_dir: Local directory to save files
            
        Returns:
            List of downloaded file paths
        """
        self.logger.info(f"Downloading data from {uri}")
        downloaded_files = []
        
        try:
            fs = AzureMachineLearningFileSystem(uri)
            
            # List all files to download
            file_paths = fs.ls()
            
            for file_path in file_paths:
                local_path = os.path.join(local_dir, os.path.basename(file_path))
                
                # Download file
                with fs.open(file_path, 'rb') as src:
                    with open(local_path, 'wb') as dst:
                        dst.write(src.read())
                
                downloaded_files.append(local_path)
                self.logger.info(f"Downloaded: {file_path} -> {local_path}")
        
        except Exception as e:
            self.logger.error(f"Error downloading files: {str(e)}")
            raise
        
        return downloaded_files
    
    def setup_data_directories(self, root_dir: str) -> Dict[str, str]:
        """
        Create necessary directories for processing.
        
        Args:
            root_dir: Root directory for data
            
        Returns:
            Dictionary of directory paths
        """
        directories = {
            'data': os.path.join(root_dir, 'data'),
            'output': os.path.join(root_dir, 'output'),
            'temp': os.path.join(root_dir, 'temp'),
            'artifacts': os.path.join(root_dir, 'artifacts')
        }
        
        for dir_name, dir_path in directories.items():
            os.makedirs(dir_path, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")
        
        return directories
    
    def run_registration_processing(self) -> Dict[str, Any]:
        """
        Run registration processing using PA processor.
        
        Returns:
            Processing results and output paths
        """
        self.logger.info("Starting registration processing")
        
        try:
            # Initialize processor with config
            processor = RegistrationProcessor(self.config)
            
            # Run processing
            processor.process()
            
            # Collect output artifacts
            output_dir = self.config.get('output_dir', 'output')
            registration_outputs = {
                'processed_files': [],
                'metrics': {},
                'status': 'completed'
            }
            
            # Identify output files created by registration processor
            output_files = [
                'df_reg_demo_this.csv',
                'df_reg_demo_last_bva.csv', 
                'df_reg_demo_last_lva.csv',
                'demographic_data_this.json',
                'demographic_data_last_bva.json',
                'demographic_data_last_lva.json'
            ]
            
            for file_name in output_files:
                file_path = os.path.join(output_dir, file_name)
                if os.path.exists(file_path):
                    registration_outputs['processed_files'].append(file_path)
                    self.logger.info(f"Registration output: {file_path}")
            
            # Log metrics
            if hasattr(processor, 'statistics'):
                registration_outputs['metrics'] = processor.statistics
            
            return registration_outputs
            
        except Exception as e:
            self.logger.error(f"Registration processing failed: {str(e)}")
            raise
    
    def run_scan_processing(self) -> Dict[str, Any]:
        """
        Run scan processing using PA processor.
        
        Returns:
            Processing results and output paths
        """
        self.logger.info("Starting scan processing")
        
        try:
            # Initialize processor with config
            processor = ScanProcessor(self.config)
            
            # Run processing
            processor.process()
            
            # Collect output artifacts
            output_dir = self.config.get('output_dir', 'output')
            scan_outputs = {
                'processed_files': [],
                'metrics': {},
                'status': 'completed'
            }
            
            # Identify output files based on configuration
            scan_output_config = self.config.get('scan_output_files', {})
            
            # Process scan data outputs
            if 'scan_data' in scan_output_config:
                for key, file_name in scan_output_config['scan_data'].items():
                    file_path = os.path.join(output_dir, 'output', file_name)
                    if os.path.exists(file_path):
                        scan_outputs['processed_files'].append(file_path)
                        self.logger.info(f"Scan output: {file_path}")
            
            # Process sessions visited outputs  
            if 'sessions_visited' in scan_output_config:
                for key, file_name in scan_output_config['sessions_visited'].items():
                    file_path = os.path.join(output_dir, 'output', file_name)
                    if os.path.exists(file_path):
                        scan_outputs['processed_files'].append(file_path)
                        self.logger.info(f"Scan output: {file_path}")
            
            return scan_outputs
            
        except Exception as e:
            self.logger.error(f"Scan processing failed: {str(e)}")
            raise
    
    def run_session_processing(self) -> Dict[str, Any]:
        """
        Run session processing using PA processor.
        
        Returns:
            Processing results and output paths
        """
        self.logger.info("Starting session processing") 
        
        try:
            # Initialize processor with config
            processor = SessionProcessor(self.config)
            
            # Run processing
            processor.process()
            
            # Collect output artifacts
            output_dir = self.config.get('output_dir', 'output')
            session_outputs = {
                'processed_files': [],
                'metrics': {},
                'status': 'completed'
            }
            
            # Identify output files
            output_files = [
                'sessions_metadata_this.csv',
                'sessions_metadata_bva.csv',
                'sessions_metadata_lva.csv',
                'streams.json',
                'stream_mappings.csv'
            ]
            
            for file_name in output_files:
                file_path = os.path.join(output_dir, file_name)
                if os.path.exists(file_path):
                    session_outputs['processed_files'].append(file_path)
                    self.logger.info(f"Session output: {file_path}")
            
            return session_outputs
            
        except Exception as e:
            self.logger.error(f"Session processing failed: {str(e)}")
            raise
    
    def prepare_outputs_for_next_step(self, outputs_dir: str) -> Dict[str, str]:
        """
        Prepare and organize outputs for next pipeline steps.
        
        Args:
            outputs_dir: Directory to store outputs
            
        Returns:
            Dictionary of output paths
        """
        self.logger.info("Preparing outputs for next pipeline steps")
        
        # Create subdirectories for different output types
        output_paths = {
            'registration': os.path.join(outputs_dir, 'registration'),
            'scan': os.path.join(outputs_dir, 'scan'),
            'session': os.path.join(outputs_dir, 'session'),
            'metadata': os.path.join(outputs_dir, 'metadata')
        }
        
        for dir_name, dir_path in output_paths.items():
            os.makedirs(dir_path, exist_ok=True)
        
        # Copy processed files to appropriate directories
        import shutil
        
        # Registration outputs
        reg_files = self.output_artifacts.get('registration', {}).get('processed_files', [])
        for file_path in reg_files:
            if os.path.exists(file_path):
                dest_path = os.path.join(output_paths['registration'], os.path.basename(file_path))
                shutil.copy2(file_path, dest_path)
                self.logger.info(f"Copied registration file: {dest_path}")
        
        # Scan outputs  
        scan_files = self.output_artifacts.get('scan', {}).get('processed_files', [])
        for file_path in scan_files:
            if os.path.exists(file_path):
                dest_path = os.path.join(output_paths['scan'], os.path.basename(file_path))
                shutil.copy2(file_path, dest_path)
                self.logger.info(f"Copied scan file: {dest_path}")
        
        # Session outputs
        session_files = self.output_artifacts.get('session', {}).get('processed_files', [])
        for file_path in session_files:
            if os.path.exists(file_path):
                dest_path = os.path.join(output_paths['session'], os.path.basename(file_path))
                shutil.copy2(file_path, dest_path)
                self.logger.info(f"Copied session file: {dest_path}")
        
        # Create metadata file
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config_file': self.config.get('config_file_path', 'unknown'),
            'incremental': self.incremental,
            'processors_run': ['registration', 'scan', 'session'],
            'output_files': {
                'registration': [os.path.basename(f) for f in reg_files],
                'scan': [os.path.basename(f) for f in scan_files],
                'session': [os.path.basename(f) for f in session_files]
            },
            'metrics': {
                'registration': self.output_artifacts.get('registration', {}).get('metrics', {}),
                'scan': self.output_artifacts.get('scan', {}).get('metrics', {}),
                'session': self.output_artifacts.get('session', {}).get('metrics', {})
            }
        }
        
        metadata_path = os.path.join(output_paths['metadata'], 'step1_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Created metadata file: {metadata_path}")
        
        return output_paths
    
    def log_mlflow_metrics(self):
        """Log metrics to MLflow for tracking."""
        try:
            # Log basic metrics
            for processor_name in ['registration', 'scan', 'session']:
                if processor_name in self.output_artifacts:
                    processor_data = self.output_artifacts[processor_name]
                    
                    # Log file counts
                    file_count = len(processor_data.get('processed_files', []))
                    mlflow.log_metric(f"{processor_name}_output_files", file_count)
                    
                    # Log processor-specific metrics
                    for metric_name, metric_value in processor_data.get('metrics', {}).items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(f"{processor_name}_{metric_name}", metric_value)
            
            # Log parameters
            mlflow.log_param("step_name", "data_preparation")
            mlflow.log_param("incremental_processing", self.incremental)
            mlflow.log_param("config_file", self.config.get('config_file_path', 'unknown'))
            
            self.logger.info("MLflow metrics logged successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to log MLflow metrics: {str(e)}")
    
    def process(self) -> Dict[str, str]:
        """
        Main processing method for the data preparation step.
        
        Returns:
            Dictionary of output paths
        """
        self.logger.info("="*60)
        self.logger.info("Starting Azure ML Data Preparation Step")
        self.logger.info("="*60)
        
        try:
            # Track processing time
            start_time = datetime.now()
            
            # Run processors based on configuration
            if self.config.get('pipeline_steps', {}).get('registration_processing', True):
                self.output_artifacts['registration'] = self.run_registration_processing()
            else:
                self.logger.info("Registration processing disabled in configuration")
            
            if self.config.get('pipeline_steps', {}).get('scan_processing', True):
                self.output_artifacts['scan'] = self.run_scan_processing()
            else:
                self.logger.info("Scan processing disabled in configuration")
            
            if self.config.get('pipeline_steps', {}).get('session_processing', True):
                self.output_artifacts['session'] = self.run_session_processing()
            else:
                self.logger.info("Session processing disabled in configuration")
            
            # Prepare outputs for next steps
            outputs_dir = os.path.join(os.getcwd(), 'outputs')
            output_paths = self.prepare_outputs_for_next_step(outputs_dir)
            
            # Log MLflow metrics
            self.log_mlflow_metrics()
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            mlflow.log_metric("step1_total_processing_time_seconds", processing_time)
            
            # Print summary
            self.logger.info("="*60)
            self.logger.info("Data Preparation Step Completed Successfully")
            self.logger.info(f"Processing time: {processing_time:.2f} seconds")
            self.logger.info(f"Output directory: {outputs_dir}")
            self.logger.info("="*60)
            
            return output_paths
            
        except Exception as e:
            self.logger.error(f"Data preparation step failed: {str(e)}")
            raise


def main(args):
    """Main function to run the data preparation step."""
    
    # Enable auto logging
    mlflow.autolog()
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize credentials and ML client if needed
        subscription_id = os.getenv("SUBSCRIPTION_ID")
        resource_group = os.getenv("RESOURCE_GROUP")
        workspace = os.getenv("AZUREML_WORKSPACE_NAME")
        
        if all([subscription_id, resource_group, workspace]):
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential, subscription_id, resource_group, workspace
            )
        
        # Download input data if URI provided
        root_dir = os.path.dirname(os.path.abspath(__file__))
        
        if args.input_uri:
            # Create data directory
            data_dir = os.path.join(root_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Download data from blob
            step = DataPreparationStep(args.config, args.incremental)
            downloaded_files = step.download_blob_data(args.input_uri, data_dir)
            
            if not downloaded_files:
                raise Exception("No files were downloaded from input URI")
        
        # Run data preparation step
        step = DataPreparationStep(args.config, args.incremental)
        output_paths = step.process()
        
        print("\n" + "="*60)
        print("DATA PREPARATION STEP SUMMARY")
        print("="*60)
        print(f"Configuration: {args.config}")
        print(f"Incremental: {args.incremental}")
        print(f"Outputs created in: {output_paths}")
        print("="*60)
        
    except Exception as e:
        print(f"JOB FAILED: {str(e)}")
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Azure ML Data Preparation Step")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--input_uri",
        type=str,
        help="URI of input data in Azure Blob Storage"
    )
    
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Run incremental processing (only new data)"
    )
    
    parser.add_argument(
        "--output_registration",
        type=str,
        help="Output path for registration data"
    )
    
    parser.add_argument(
        "--output_scan",
        type=str,
        help="Output path for scan data"
    )
    
    parser.add_argument(
        "--output_session",
        type=str,
        help="Output path for session data"
    )
    
    parser.add_argument(
        "--output_metadata",
        type=str,
        help="Output path for metadata"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)