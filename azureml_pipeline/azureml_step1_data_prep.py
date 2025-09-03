#!/usr/bin/env python
"""
Azure ML Pipeline Step 1: Data Preparation
Processes registration, scan, and session data for Personal Agendas pipeline.
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

# Import with fallback

from PA.registration_processor import RegistrationProcessor
from PA.scan_processor import ScanProcessor
from PA.session_processor import SessionProcessor
from PA.utils.config_utils import load_config
from PA.utils.logging_utils import setup_logging
from PA.utils.keyvault_utils import ensure_env_file, KeyVaultManager



class DataPreparationStep:
    """Azure ML Data Preparation Step for Personal Agendas pipeline."""
    
    def __init__(self, config_path: str, incremental: bool = False, use_keyvault: bool = True):
        """
        Initialize the Data Preparation Step.
        
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
    
    def _is_azure_ml_environment(self) -> bool:
        """Check if running in Azure ML environment."""
        # Azure ML sets specific environment variables
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
            
            # Ensure .env file exists from Key Vault
            env_path = os.path.join(project_root, "PA", "keys", ".env")
            if ensure_env_file(keyvault_name, env_path):
                self.logger.info("Successfully loaded secrets from Key Vault")
            else:
                self.logger.warning("Could not load secrets from Key Vault, will try environment variables")
                
        except Exception as e:
            self.logger.error(f"Error loading secrets from Key Vault: {e}")
            self.logger.info("Falling back to environment variables or existing .env file")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(root_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'data_prep_{timestamp}.log')
        
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
        self.logger.info(f"Loading configuration from {config_path}")
        
        # Load base configuration
        config = load_config(config_path)
        
        # CRITICAL FIX: Ensure env_file path is absolute
        # This fixes the SessionProcessor .env loading issue
        if 'env_file' in config:
            # If env_file is a relative path, make it absolute relative to project root
            env_file_path = config['env_file']
            if not os.path.isabs(env_file_path):
                # The env file is expected to be at PA/keys/.env
                abs_env_path = os.path.join(project_root, 'PA', env_file_path)
                # Also try without PA prefix if it's already included
                if not os.path.exists(abs_env_path):
                    abs_env_path = os.path.join(project_root, env_file_path)
                
                if os.path.exists(abs_env_path):
                    config['env_file'] = abs_env_path
                    self.logger.info(f"Updated env_file path to absolute: {abs_env_path}")
                else:
                    # If .env file doesn't exist, create it or use environment variables
                    self.logger.warning(f"Environment file not found at {abs_env_path}")
                    # Try to create a temporary env file from Azure environment variables
                    self._create_temp_env_file(config)
        
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
    
    def copy_outputs_to_azure_ml(self, output_paths: Dict[str, str]) -> None:
        """
        Copy processor outputs to Azure ML output directories for Step 2.
        Following Azure ML SDK v2 pattern for data passing between pipeline steps.
        
        Args:
            output_paths: Dictionary of Azure ML output paths from argparse
        """
        self.logger.info("Copying outputs to Azure ML directories for next steps")
        
        event_name = self.config.get('event', {}).get('name', 'ecomm')
        
        # Map local files to Azure ML outputs
        # These mappings ensure Step 2 finds the files with expected names
        file_mappings = {
            'output_registration': [
                # Registration outputs from data/{event}/output/
                (f'data/{event_name}/output/df_reg_demo_this.csv', 'df_reg_demo_this.csv'),
                (f'data/{event_name}/output/df_reg_demo_last_bva.csv', 'df_reg_demo_last_bva.csv'),
                (f'data/{event_name}/output/df_reg_demo_last_lva.csv', 'df_reg_demo_last_lva.csv'),
            ],
            'output_scan': [
                # Scan outputs - note the name mappings for ecomm
                (f'data/{event_name}/output/sessions_visited_last_bva.csv', 'sessions_visited_last_bva.csv'),
                (f'data/{event_name}/output/sessions_visited_last_lva.csv', 'sessions_visited_last_lva.csv'),
                # Also try the alternate names
                (f'data/{event_name}/output/sessions_with_demo_ecomm.csv', 'scan_last_filtered_valid_cols_ecomm.csv'),
                (f'data/{event_name}/output/sessions_with_demo_tfm.csv', 'scan_last_filtered_valid_cols_tfm.csv'),
                (f'data/{event_name}/output/scan_bva_past.csv', 'scan_bva_past.csv'),
                (f'data/{event_name}/output/scan_lva_past.csv', 'scan_lva_past.csv'),
            ],
            'output_session': [
                # Session outputs from data/{event}/output/
                (f'data/{event_name}/output/session_this_filtered_valid_cols.csv', 'session_this_filtered_valid_cols.csv'),
                (f'data/{event_name}/output/session_last_filtered_valid_cols_bva.csv', 'session_last_filtered_valid_cols_bva.csv'),
                (f'data/{event_name}/output/session_last_filtered_valid_cols_lva.csv', 'session_last_filtered_valid_cols_lva.csv'),
                (f'data/{event_name}/output/streams.json', 'streams.json'),
                (f'data/{event_name}/output/streams.csv', 'streams.csv'),
            ]
        }
        
        # Copy files to appropriate Azure ML output directories
        for output_key, files in file_mappings.items():
            if output_key in output_paths and output_paths[output_key]:
                output_path = output_paths[output_key]
                os.makedirs(output_path, exist_ok=True)
                self.logger.info(f"Copying {output_key} files to: {output_path}")
                
                for src_relative, dest_name in files:
                    # Try the primary location
                    src_path = os.path.join(root_dir, src_relative)
                    
                    if os.path.exists(src_path):
                        dest_path = os.path.join(output_path, dest_name)
                        shutil.copy2(src_path, dest_path)
                        self.logger.info(f"  Copied {dest_name}")
                    else:
                        # Try alternate location (directly in data/output)
                        alt_src = os.path.join(root_dir, 'data', 'output', dest_name)
                        if os.path.exists(alt_src):
                            dest_path = os.path.join(output_path, dest_name)
                            shutil.copy2(alt_src, dest_path)
                            self.logger.info(f"  Copied {dest_name} from alt location")
                        else:
                            self.logger.warning(f"  File not found: {src_relative}")

    def _create_temp_env_file(self, config: Dict[str, Any]) -> None:
        """
        Create a temporary .env file from Azure environment variables.
        This is used when the .env file is not found in the expected location.
        
        Args:
            config: Configuration dictionary
        """
        # Create PA/keys directory structure if it doesn't exist
        keys_dir = os.path.join(project_root, 'PA', 'keys')
        os.makedirs(keys_dir, exist_ok=True)
        
        # Create the .env file in the expected location
        temp_env_path = os.path.join(keys_dir, '.env')
        
        # Check for environment variables that might be set in Azure ML
        env_vars = {
            'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', ''),
            'AZURE_API_KEY': os.environ.get('AZURE_API_KEY', ''),
            'AZURE_ENDPOINT': os.environ.get('AZURE_ENDPOINT', ''),
            'AZURE_DEPLOYMENT': os.environ.get('AZURE_DEPLOYMENT', ''),
            'AZURE_API_VERSION': os.environ.get('AZURE_API_VERSION', ''),
            'NEO4J_URI': os.environ.get('NEO4J_URI', ''),
            'NEO4J_USERNAME': os.environ.get('NEO4J_USERNAME', ''),
            'NEO4J_PASSWORD': os.environ.get('NEO4J_PASSWORD', '')
        }
        
        # Write environment variables to temp file
        with open(temp_env_path, 'w') as f:
            for key, value in env_vars.items():
                if value:  # Only write non-empty values
                    f.write(f"{key}={value}\n")
        
        # Update config to use the absolute path
        config['env_file'] = temp_env_path
        self.logger.info(f"Created temporary environment file at {temp_env_path}")

    def download_blob_data(self, uri_or_path: str, local_dir: str) -> List[str]:
        """
        Download data from Azure Blob Storage or copy from mounted path.
        Maps files to expected PA pipeline structure.
        
        Args:
            uri_or_path: Azure blob URI or mounted local path
            local_dir: Local directory to save files (data/)
            
        Returns:
            List of downloaded/copied file paths
        """
        self.logger.info(f"Processing input data from: {uri_or_path}")
        downloaded_files = []
        
        # Get event name from config to determine target folder
        event_name = self.config.get('event', {}).get('name', 'ecomm')
        event_folder = os.path.join(local_dir, event_name)
        os.makedirs(event_folder, exist_ok=True)
        
        # Define file mappings based on the configuration
        # Maps source filename patterns to expected locations
        file_mappings = self._get_file_mappings(event_name)
        
        try:
            # Check if it's a local path (mounted by Azure ML) or a URI
            if os.path.exists(uri_or_path) and os.path.isdir(uri_or_path):
                # It's a mounted path - copy files with proper mapping
                self.logger.info(f"Input is a mounted directory: {uri_or_path}")
                
                # List all files in the mounted directory
                for root, dirs, files in os.walk(uri_or_path):
                    for file in files:
                        source_path = os.path.join(root, file)
                        
                        # Map the file to its expected destination
                        dest_path = self._map_file_path(
                            file, source_path, event_folder, file_mappings
                        )
                        
                        if dest_path:
                            # Create destination directory if needed
                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                            
                            # Copy file
                            shutil.copy2(source_path, dest_path)
                            downloaded_files.append(dest_path)
                            self.logger.info(f"Copied: {source_path} -> {dest_path}")
                        else:
                            self.logger.warning(f"No mapping found for file: {file}")
                        
            elif uri_or_path.startswith('azureml://'):
                # It's an Azure ML URI - use AzureMachineLearningFileSystem
                self.logger.info(f"Input is an Azure ML URI: {uri_or_path}")
                
                fs = AzureMachineLearningFileSystem(uri_or_path)
                
                # List all files to download
                file_paths = fs.ls()
                
                for file_path in file_paths:
                    file_name = os.path.basename(file_path)
                    
                    # Map the file to its expected destination
                    dest_path = self._map_file_path(
                        file_name, file_path, event_folder, file_mappings
                    )
                    
                    if dest_path:
                        # Create destination directory if needed
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        
                        # Download file
                        with fs.open(file_path, 'rb') as src:
                            with open(dest_path, 'wb') as dst:
                                dst.write(src.read())
                        
                        downloaded_files.append(dest_path)
                        self.logger.info(f"Downloaded: {file_path} -> {dest_path}")
                    else:
                        self.logger.warning(f"No mapping found for file: {file_name}")
            else:
                # Unknown format
                raise ValueError(f"Input path format not recognized: {uri_or_path}")
                
        except Exception as e:
            self.logger.error(f"Error processing input data: {str(e)}")
            raise
        
        self.logger.info(f"Successfully processed {len(downloaded_files)} files")
        return downloaded_files
    
    def _get_file_mappings(self, event_name: str) -> Dict[str, str]:
        """
        Get file mappings based on event type.
        Maps source filenames (or patterns) to expected paths in data/ folder.
        
        Args:
            event_name: Name of the event (ecomm, vet, etc.)
            
        Returns:
            Dictionary of filename patterns to target paths
        """
        # These mappings should align with what's expected in config yaml files
        if event_name == 'ecomm':
            return {
                # Registration files
                '20250818_registration_ECE_TFM_24_25.json': 'data/ecomm/20250818_registration_ECE_TFM_24_25.json',
                '20250722_registration_TFM24.json': 'data/ecomm/20250722_registration_TFM24.json',
                # Demographic files
                '20250818_demographics_ECE_TFM_24_25.json': 'data/ecomm/20250818_demographics_ECE_TFM_24_25.json',
                '20250722_demographics_TFM24.json': 'data/ecomm/20250722_demographics_TFM24.json',
                # Session files
                'ECE_TFM_25_session_export.csv': 'data/ecomm/ECE_TFM_25_session_export.csv',
                'ECE25_session_export.csv': 'data/ecomm/ECE25_session_export.csv',
                'ECE24_session_export.csv': 'data/ecomm/ECE24_session_export.csv',
                'TFM24_session_export.csv': 'data/ecomm/TFM24_session_export.csv',
                # Scan files
                'ece2024 seminar scans reference.csv': 'data/ecomm/ece2024 seminar scans reference.csv',
                'ece2024 seminar scans.csv': 'data/ecomm/ece2024 seminar scans.csv',
                'tfm2024 seminar scans reference.csv': 'data/ecomm/tfm2024 seminar scans reference.csv',
                'tfm2024 seminar scans.csv': 'data/ecomm/tfm2024 seminar scans.csv',
            }
        elif event_name == 'vet':
            return {
                # Registration files
                '20250818_registration_BVA_LVA_24_25.json': 'data/vet/20250818_registration_BVA_LVA_24_25.json',
                '20250722_registration_LVS24.json': 'data/vet/20250722_registration_LVS24.json',
                # Demographic files
                '20250818_demographics_BVA_LVA_24_25.json': 'data/vet/20250818_demographics_BVA_LVA_24_25.json',
                '20250722_demographics_LVS24.json': 'data/vet/20250722_demographics_LVS24.json',
                # Session files
                'BVA25_session_export.csv': 'data/vet/BVA25_session_export.csv',
                'BVA24_session_export.csv': 'data/vet/BVA24_session_export.csv',
                'LVS24_session_export.csv': 'data/vet/LVS24_session_export.csv',
                # Scan files
                'bva2024 seminar scans reference.csv': 'data/vet/bva2024 seminar scans reference.csv',
                'bva2024 seminar scans.csv': 'data/vet/bva2024 seminar scans.csv',
                'lva2024 seminar scans reference.csv': 'data/vet/lva2024 seminar scans reference.csv',
                'lva2024 seminar scans.csv': 'data/vet/lva2024 seminar scans.csv',
            }
        else:
            # For other events, just copy to event folder maintaining structure
            return {}
    
    def _map_file_path(
        self, 
        filename: str, 
        source_path: str, 
        event_folder: str, 
        file_mappings: Dict[str, str]
    ) -> Optional[str]:
        """
        Map a source file to its expected destination path.
        
        Args:
            filename: Name of the file
            source_path: Full source path
            event_folder: Target event folder (data/ecomm or data/vet)
            file_mappings: Dictionary of filename mappings
            
        Returns:
            Destination path or None if no mapping found
        """
        # First check exact filename match
        if filename in file_mappings:
            # Return the mapped path relative to project root
            return os.path.join(
                os.path.dirname(os.path.dirname(event_folder)),  # Go up to project root
                file_mappings[filename]
            )
        
        # Check if filename contains any mapping key (partial match)
        for pattern, target_path in file_mappings.items():
            if pattern in filename or filename in pattern:
                return os.path.join(
                    os.path.dirname(os.path.dirname(event_folder)),  # Go up to project root
                    target_path
                )
        
        # If no specific mapping, place in event folder maintaining any subdirectory structure
        # This handles any additional files not explicitly mapped
        rel_path = os.path.relpath(source_path, os.path.dirname(source_path))
        return os.path.join(event_folder, rel_path)
    
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
        self.logger.info("Starting Registration Processing")
        
        # Store the original working directory
        original_cwd = os.getcwd()
        self.logger.info(f"Current working directory: {original_cwd}")
        
        # Change to the azureml_pipeline directory where data files are located
        os.chdir(root_dir)
        self.logger.info(f"Changed working directory to: {root_dir}")
        
        try:
            processor = RegistrationProcessor(self.config)
            
            # Check if we should skip this processor
            if self.config.get('processors', {}).get('registration_processing', {}).get('enabled', True):
                # Call process without arguments - PA processors don't accept incremental parameter
                processor.process()
                
                # Collect output information
                output_dir = self.config.get('output_dir', 'output')
                output_files = []
                
                # Check for expected output files
                expected_files = [
                    'df_reg_demo_this.csv',
                    'df_reg_demo_last_bva.csv', 
                    'df_reg_demo_last_lva.csv',
                    'demographic_data_this.json',
                    'demographic_data_last_bva.json',
                    'demographic_data_last_lva.json'
                ]
                
                for filename in expected_files:
                    filepath = os.path.join(output_dir, filename)
                    if os.path.exists(filepath):
                        output_files.append(filepath)
                        self.logger.info(f"Found output file: {filepath}")
                
                result = {
                    'status': 'success',
                    'output_files': output_files,
                    'output_dir': output_dir
                }
                
                self.logger.info(f"Registration processing completed successfully")
                return result
            else:
                self.logger.info("Registration processing is disabled in config")
                return {'status': 'skipped', 'reason': 'disabled in config'}
                
        except Exception as e:
            self.logger.error(f"Registration processing failed: {str(e)}")
            raise
        finally:
            # Always restore the original working directory
            os.chdir(original_cwd)
            self.logger.info(f"Restored working directory to: {original_cwd}")
    
    def run_scan_processing(self) -> Dict[str, Any]:
        """
        Run scan processing using PA processor.
        
        Returns:
            Processing results and output paths
        """
        self.logger.info("Starting Scan Processing")
        
        # Store the original working directory
        original_cwd = os.getcwd()
        self.logger.info(f"Current working directory: {original_cwd}")
        
        # Change to the azureml_pipeline directory where data files are located
        os.chdir(root_dir)
        self.logger.info(f"Changed working directory to: {root_dir}")
        
        try:
            processor = ScanProcessor(self.config)
            
            # Check if we should skip this processor
            if self.config.get('processors', {}).get('scan_processing', {}).get('enabled', True):
                # Call process without arguments - PA processors don't accept incremental parameter
                processor.process()
                
                # Collect output information
                output_dir = self.config.get('output_dir', 'output')
                output_files = []
                
                # Check for expected output files
                expected_files = [
                    'sessions_visited_last_bva.csv',
                    'sessions_visited_last_lva.csv',
                    'scan_lva_past',
                    'scan_bva_past'
                ]
                
                for filename in expected_files:
                    filepath = os.path.join(output_dir, filename)
                    if os.path.exists(filepath):
                        output_files.append(filepath)
                        self.logger.info(f"Found output file: {filepath}")
                
                result = {
                    'status': 'success',
                    'output_files': output_files,
                    'output_dir': output_dir
                }
                
                self.logger.info(f"Scan processing completed successfully")
                return result
            else:
                self.logger.info("Scan processing is disabled in config")
                return {'status': 'skipped', 'reason': 'disabled in config'}
                
        except Exception as e:
            self.logger.error(f"Scan processing failed: {str(e)}")
            raise
        finally:
            # Always restore the original working directory
            os.chdir(original_cwd)
            self.logger.info(f"Restored working directory to: {original_cwd}")
    
    def run_session_processing(self) -> Dict[str, Any]:
        """
        Run session processing using PA processor.
        
        Returns:
            Processing results and output paths
        """
        self.logger.info("Starting Session Processing")
        
        # Store the original working directory
        original_cwd = os.getcwd()
        self.logger.info(f"Current working directory: {original_cwd}")
        
        # Change to the azureml_pipeline directory where data files are located
        os.chdir(root_dir)
        self.logger.info(f"Changed working directory to: {root_dir}")
        
        try:
            # CRITICAL: Ensure the config has the correct absolute path for env_file
            # This was already done in _load_configuration, but let's verify
            if 'env_file' in self.config:
                env_file_path = self.config['env_file']
                if not os.path.exists(env_file_path):
                    self.logger.warning(f"Env file not found at {env_file_path}, checking alternatives...")
                    
                    # Try different possible locations
                    possible_paths = [
                        os.path.join(original_cwd, 'PA', 'keys', '.env'),
                        os.path.join(root_dir, 'PA', 'keys', '.env'),
                        os.path.join(project_root, 'PA', 'keys', '.env'),
                        os.path.join(os.getcwd(), 'PA', 'keys', '.env'),
                        os.path.join(os.getcwd(), 'keys', '.env'),
                        'PA/keys/.env',
                        'keys/.env'
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            self.config['env_file'] = os.path.abspath(path)
                            self.logger.info(f"Found env file at: {path}")
                            break
                    else:
                        # If no .env file found, create one from environment variables
                        self._create_temp_env_file(self.config)
            
            # Initialize the processor with the updated config
            processor = SessionProcessor(self.config)
            
            # Check if we should skip this processor
            if self.config.get('processors', {}).get('session_processing', {}).get('enabled', True):
                # Call process without arguments - PA processors don't accept incremental parameter
                processor.process()
                
                # Collect output information
                output_dir = self.config.get('output_dir', 'output')
                output_files = []
                
                # Check for expected output files based on event type
                event_name = self.config.get('event', {}).get('name', 'ecomm')
                
                if event_name == 'ecomm':
                    expected_files = [
                        'session_this_filtered_valid_cols.csv',
                        'session_last_filtered_valid_cols_bva.csv',  # Using bva suffix for ecomm
                        'session_last_filtered_valid_cols_lva.csv',  # Using lva suffix for tfm
                        'streams.json'
                    ]
                else:  # vet/bva
                    expected_files = [
                        'session_this_filtered_valid_cols.csv',
                        'session_last_filtered_valid_cols_bva.csv',
                        'session_last_filtered_valid_cols_lva.csv',
                        'streams.json'
                    ]
                
                # Look for output files
                output_path = os.path.join(output_dir, 'output')
                for filename in expected_files:
                    filepath = os.path.join(output_path, filename)
                    if os.path.exists(filepath):
                        output_files.append(filepath)
                        self.logger.info(f"Found output file: {filepath}")
                    else:
                        self.logger.warning(f"Expected output file not found: {filepath}")
                
                result = {
                    'status': 'success',
                    'output_files': output_files,
                    'output_dir': output_dir
                }
                
                self.logger.info(f"Session processing completed successfully")
                return result
            else:
                self.logger.info("Session processing is disabled in config")
                return {'status': 'skipped', 'reason': 'disabled in config'}
                
        except Exception as e:
            self.logger.error(f"Session processing failed: {str(e)}")
            raise
        finally:
            # Always restore the original working directory
            os.chdir(original_cwd)
            self.logger.info(f"Restored working directory to: {original_cwd}")
    

    def save_outputs(self, results: Dict[str, Any], output_paths: Dict[str, str]) -> None:
        """
        Save processing results to Azure ML output locations.
        Ensures files are available for Step 2 by properly copying to mounted output paths.
        
        Args:
            results: Processing results from all processors
            output_paths: Dictionary of output paths from argparse (Azure ML mounted paths)
        """
        self.logger.info("Saving outputs to Azure ML paths")
        self.logger.info(f"Output paths provided: {output_paths}")
        
        event_name = self.config.get('event', {}).get('name', 'ecomm')
        
        # Map processor outputs to the files Step 2 expects
        # The key is the output mount, the value is list of (source_file, dest_name) tuples
        file_mappings = {
            'output_registration': [
                # These files are created by registration_processor in data/{event}/output/
                (f'data/{event_name}/output/df_reg_demo_this.csv', 'df_reg_demo_this.csv'),
                (f'data/{event_name}/output/df_reg_demo_last_bva.csv', 'df_reg_demo_last_bva.csv'),
                (f'data/{event_name}/output/df_reg_demo_last_lva.csv', 'df_reg_demo_last_lva.csv'),

            ],
            'output_scan': [
                # Scan processor creates these files
                # For ecomm event:
                (f'data/{event_name}/output/sessions_visited_last_bva.csv', 'sessions_visited_last_bva.csv'),
                (f'data/{event_name}/output/sessions_visited_last_lva.csv', 'sessions_visited_last_lva.csv'),
                (f'data/{event_name}/output/scan_lva_past.csv', 'scan_lva_past.csv'),
                (f'data/{event_name}/output/scan_bva_past.csv', 'scan_bva_past.csv'),

            ],
            'output_session': [
                # Session processor outputs
                (f'data/{event_name}/output/session_this_filtered_valid_cols.csv', 'session_this_filtered_valid_cols.csv'),
                (f'data/{event_name}/output/session_last_filtered_valid_cols_bva.csv', 'session_last_filtered_valid_cols_bva.csv'),
                (f'data/{event_name}/output/session_last_filtered_valid_cols_lva.csv', 'session_last_filtered_valid_cols_lva.csv'),
                (f'data/{event_name}/output/streams.json', 'streams.json'),
                (f'data/{event_name}/output/streams.csv', 'streams.csv'),
                # Cache file if exists
                (f'data/{event_name}/output/streams_cache.json', 'streams_cache.json'),
            ]
        }
        
        # Process each output type
        total_files_copied = 0
        for output_key, file_list in file_mappings.items():
            if output_key in output_paths and output_paths[output_key]:
                output_path = output_paths[output_key]
                
                # CRITICAL: Ensure the output directory exists
                # Azure ML mounts the path but we still need to create subdirectories if needed
                try:
                    os.makedirs(output_path, exist_ok=True)
                    self.logger.info(f"\nProcessing {output_key}:")
                    self.logger.info(f"  Output path: {output_path}")
                except Exception as e:
                    self.logger.error(f"  Failed to create output directory: {e}")
                    continue
                
                files_copied_for_output = 0
                
                for source_file, dest_name in file_list:
                    # Build full source path
                    full_source = os.path.join(root_dir, source_file)
                    
                    # Try multiple possible locations
                    possible_sources = [
                        full_source,  # Primary expected location
                        os.path.join(root_dir, 'data', 'output', dest_name),  # Fallback to data/output
                        os.path.join(root_dir, 'data', 'output', os.path.basename(source_file)),  # Original filename in output
                        os.path.join(root_dir, 'azureml_pipeline', source_file),  # In case of relative path issues
                    ]
                    
                    file_copied = False
                    for src in possible_sources:
                        if os.path.exists(src) and os.path.isfile(src):
                            dest_path = os.path.join(output_path, dest_name)
                            try:
                                shutil.copy2(src, dest_path)
                                self.logger.info(f"  ✓ Copied: {dest_name} ({os.path.getsize(src)} bytes)")
                                files_copied_for_output += 1
                                file_copied = True
                                break
                            except Exception as e:
                                self.logger.error(f"  ✗ Failed to copy {dest_name}: {e}")
                    
                    if not file_copied:
                        # Log as warning but don't fail - some files might be optional
                        self.logger.warning(f"  ⚠ Not found: {dest_name} (searched {len(possible_sources)} locations)")
                
                total_files_copied += files_copied_for_output
                self.logger.info(f"  Summary: {files_copied_for_output} files copied to {output_key}")
        
        self.logger.info(f"\nTotal files copied to outputs: {total_files_copied}")
        
        # Save metadata
        if 'output_metadata' in output_paths and output_paths['output_metadata']:
            metadata_path = output_paths['output_metadata']
            try:
                os.makedirs(metadata_path, exist_ok=True)
                
                # Create comprehensive metadata
                metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'config_path': self.config_path,
                    'event_name': event_name,
                    'incremental': self.incremental,
                    'processors_run': [],
                    'results': results,
                    'files_copied': total_files_copied,
                    'output_locations': {
                        'registration': output_paths.get('output_registration', ''),
                        'scan': output_paths.get('output_scan', ''),
                        'session': output_paths.get('output_session', ''),
                    }
                }
                
                # List successful processors
                for processor_name, result in results.items():
                    if result.get('status') == 'success':
                        metadata['processors_run'].append(processor_name)
                        # Add output file info if available
                        if 'output_files' in result:
                            metadata[f'{processor_name}_outputs'] = result['output_files']
                
                # Save metadata JSON
                metadata_file = os.path.join(metadata_path, 'metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.logger.info(f"Saved metadata to {metadata_file}")
                
                # Also create a summary text file
                summary_file = os.path.join(metadata_path, 'step1_summary.txt')
                with open(summary_file, 'w') as f:
                    f.write("Data Preparation Step Summary\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Configuration: {self.config_path}\n")
                    f.write(f"Event Type: {event_name}\n")
                    f.write(f"Incremental: {self.incremental}\n")
                    f.write(f"Files Copied to Outputs: {total_files_copied}\n")
                    f.write("\nProcessors Run:\n")
                    for processor in metadata['processors_run']:
                        f.write(f"  - {processor}\n")
                    f.write("\nOutput Locations:\n")
                    for key, path in metadata['output_locations'].items():
                        if path:
                            f.write(f"  - {key}: {path}\n")
                            # List files in that output
                            if os.path.exists(path):
                                files = os.listdir(path)
                                for file in files[:10]:  # List first 10 files
                                    f.write(f"      • {file}\n")
                
                self.logger.info(f"Saved summary to {summary_file}")
                
            except Exception as e:
                self.logger.error(f"Failed to save metadata: {e}")
        
        # CRITICAL: Verify outputs are accessible
        self.logger.info("\n" + "="*50)
        self.logger.info("VERIFYING OUTPUT ACCESSIBILITY")
        self.logger.info("="*50)
        
        for output_key in ['output_registration', 'output_scan', 'output_session']:
            if output_key in output_paths and output_paths[output_key]:
                path = output_paths[output_key]
                if os.path.exists(path) and os.path.isdir(path):
                    files = os.listdir(path)
                    self.logger.info(f"{output_key}: ✓ Accessible ({len(files)} files)")
                    if files:
                        self.logger.info(f"  Files: {', '.join(files[:5])}")  # Show first 5
                else:
                    self.logger.error(f"{output_key}: ✗ NOT ACCESSIBLE - Step 2 will fail!")
        
        self.logger.info("="*50)
    
    def process(self) -> Dict[str, Any]:
        """
        Run the complete data preparation step.
        
        Returns:
            Dictionary containing results from all processors
        """
        self.logger.info("="*60)
        self.logger.info("Starting Data Preparation Step")
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Incremental: {self.incremental}")
        self.logger.info("="*60)
        
        # Setup directories using module-level root_dir
        directories = self.setup_data_directories(root_dir)
        
        # Initialize results
        results = {
            'registration': {},
            'scan': {},
            'session': {}
        }
        
        # Run processors in sequence
        if self.config.get('processors', {}).get('registration_processing', {}).get('enabled', True):
            self.logger.info("\n" + "="*40)
            self.logger.info("REGISTRATION PROCESSING")
            self.logger.info("="*40)
            results['registration'] = self.run_registration_processing()
        
        if self.config.get('processors', {}).get('scan_processing', {}).get('enabled', True):
            self.logger.info("\n" + "="*40)
            self.logger.info("SCAN PROCESSING")
            self.logger.info("="*40)
            results['scan'] = self.run_scan_processing()
        
        if self.config.get('processors', {}).get('session_processing', {}).get('enabled', True):
            self.logger.info("\n" + "="*40)
            self.logger.info("SESSION PROCESSING")
            self.logger.info("="*40)
            results['session'] = self.run_session_processing()
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Data Preparation Step Completed Successfully")
        self.logger.info("="*60)
        
        return results


def main(args):
    """Main entry point for Azure ML step."""
    print("\n" + "="*60)
    print("AZURE ML DATA PREPARATION STEP")
    print("="*60)
    
    # Enable auto logging
    # mlflow.autolog()
    
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
        
        # Initialize the step
        step = DataPreparationStep(args.config, args.incremental)
        
        # Download input data if URI provided
        if args.input_uri:
            # Create data directory
            data_dir = os.path.join(root_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Download/copy data
            downloaded_files = step.download_blob_data(args.input_uri, data_dir)
            
            if not downloaded_files:
                raise Exception("No files were downloaded from input URI")
            
            print(f"Successfully processed {len(downloaded_files)} input files")
        
        # Run data preparation step
        results = step.process()
        
        # Save outputs to Azure ML locations
        output_paths = {
            'output_registration': args.output_registration,
            'output_scan': args.output_scan,
            'output_session': args.output_session,
            'output_metadata': args.output_metadata
        }
        
        step.save_outputs(results, output_paths)
        
        print("\n" + "="*60)
        print("DATA PREPARATION STEP SUMMARY")
        print("="*60)
        print(f"Configuration: {args.config}")
        print(f"Incremental: {args.incremental}")
        print(f"Results:")
        for processor, result in results.items():
            status = result.get('status', 'unknown')
            print(f"  - {processor}: {status}")
        print("="*60)
        
    except Exception as e:
        print(f"JOB FAILED: {str(e)}")
        traceback.print_exc()
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
        help="URI or path of input data (can be Azure URI or mounted path)"
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