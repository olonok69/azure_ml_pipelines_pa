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
from typing import Dict, List, Any, Optional, Union, Tuple

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
from PA.utils.app_insights import configure_app_insights



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

    def _output_source_roots(self, event_name: str) -> List[Path]:
        roots: List[Path] = []
        event_root = Path(root_dir) / 'data' / event_name / 'output'
        if event_root.exists():
            roots.append(event_root)

        generic_root = Path(root_dir) / 'data' / 'output'
        if generic_root.exists():
            roots.append(generic_root)

        return roots

    def _collect_output_files(self, event_name: str) -> Dict[str, Path]:
        file_index: Dict[str, Path] = {}
        for source_root in self._output_source_roots(event_name):
            for artifact in source_root.rglob('*'):
                if artifact.is_file():
                    key = artifact.name.lower()
                    if key not in file_index:
                        file_index[key] = artifact
        return file_index

    def _categorize_output_file(self, filename: str) -> List[str]:
        name = filename.lower()
        categories: List[str] = []

        if (
            name.startswith('registration_')
            or name.startswith('df_reg_')
            or name.startswith('demographic_')
            or 'registration' in name
        ):
            categories.append('registration')

        if (
            name.startswith('scan_')
            or name.startswith('sessions_visited')
            or 'scan_' in name
        ):
            categories.append('scan')

        if (
            name.startswith('session_')
            or name.endswith('_session_export.csv')
            or 'stream' in name
            or 'teatre' in name
        ):
            categories.append('session')

        return categories
    
    def copy_outputs_to_azure_ml(self, output_paths: Dict[str, str]) -> None:
        """
        Copy processor outputs to Azure ML output directories for Step 2.
        Following Azure ML SDK v2 pattern for data passing between pipeline steps.
        
        Args:
            output_paths: Dictionary of Azure ML output paths from argparse
        """
        self.logger.info("Copying outputs to Azure ML directories for next steps")
        
        event_name = self.config.get('event', {}).get('name', 'ecomm')

        file_index = self._collect_output_files(event_name)
        if not file_index:
            self.logger.warning(
                "No processor outputs discovered for event '%s'; nothing to copy to Azure ML outputs",
                event_name,
            )
            return

        category_to_output = {
            'registration': 'output_registration',
            'scan': 'output_scan',
            'session': 'output_session',
        }

        category_counts = {key: 0 for key in category_to_output.keys()}
        unclassified: List[Path] = []

        for file_key, source_path in file_index.items():
            categories = self._categorize_output_file(source_path.name)
            if not categories:
                unclassified.append(source_path)
                continue

            for category in categories:
                output_key = category_to_output.get(category)
                destination_dir = output_paths.get(output_key)
                if not destination_dir:
                    continue

                Path(destination_dir).mkdir(parents=True, exist_ok=True)
                dest_path = Path(destination_dir) / source_path.name
                shutil.copy2(source_path, dest_path)
                category_counts[category] += 1
                self.logger.info(
                    "Copied %s to %s",
                    source_path.name,
                    destination_dir,
                )

        if unclassified:
            fallback_dir = output_paths.get('output_session') or output_paths.get('output_scan')
            if fallback_dir:
                Path(fallback_dir).mkdir(parents=True, exist_ok=True)
                for source_path in unclassified:
                    shutil.copy2(source_path, Path(fallback_dir) / source_path.name)
                self.logger.info(
                    "Copied %s unclassified files to %s",
                    len(unclassified),
                    fallback_dir,
                )
            else:
                self.logger.warning(
                    "%s files could not be categorized and no fallback output was available",
                    len(unclassified),
                )

        for category, count in category_counts.items():
            self.logger.info("Category '%s' -> %s files", category, count)

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

    def _activate_vet_specific_functions(self, processor: RegistrationProcessor) -> bool:
        """Apply veterinary-specific overrides when the config targets a vet show."""
        event_cfg = self.config.get('event', {}) or {}
        main_event_name = str(
            event_cfg.get('main_event_name')
            or event_cfg.get('name')
            or ''
        ).lower()

        if main_event_name not in {'bva', 'lva'}:
            self.logger.info(
                "Registration processor running with generic logic for event '%s'",
                main_event_name or 'unknown',
            )
            return False

        try:
            from utils import vet_specific_functions  # type: ignore
        except ImportError:
            from PA.utils import vet_specific_functions  # type: ignore

        self.logger.info(
            "Applying veterinary-specific registration overrides for event '%s'",
            main_event_name,
        )

        try:
            vet_specific_functions.add_vet_specific_methods(processor)
            if vet_specific_functions.verify_vet_functions_applied(processor):
                processor.logger.info("Veterinary-specific functions active for this run")
                return True

            self.logger.error("Failed to verify veterinary-specific overrides on processor")
        except Exception as exc:
            self.logger.error("Error activating veterinary-specific functions: %s", exc, exc_info=True)

        return False

    def download_blob_data(self, uri_or_path: str, local_dir: str) -> List[str]:
        """
        Download data from Azure Blob Storage or copy from mounted path.
        Maps files to expected PA pipeline structure using config-defined inputs.
        
        Args:
            uri_or_path: Azure blob URI or mounted local path
            local_dir: Local directory to save files (data/)
            
        Returns:
            List of downloaded/copied file paths
        """
        self.logger.info(f"Processing input data from: {uri_or_path}")
        downloaded_files: List[str] = []
        
        event_name = self.config.get('event', {}).get('name', 'ecomm')
        event_folder = os.path.join(local_dir, event_name)
        os.makedirs(event_folder, exist_ok=True)
        
        file_index, path_lookup = self._build_config_file_index()
        expected_total = sum(len(entries) for entries in file_index.values())
        if expected_total:
            self.logger.info(
                f"Tracking {expected_total} config-defined input files before copying payload"
            )
        else:
            self.logger.warning(
                "No config-driven input files detected; all files will be placed directly under the event folder"
            )
        
        try:
            if os.path.exists(uri_or_path) and os.path.isdir(uri_or_path):
                self.logger.info(f"Input is a mounted directory: {uri_or_path}")
                for root, _, files in os.walk(uri_or_path):
                    for filename in files:
                        source_path = os.path.join(root, filename)
                        dest_path, matched_entry = self._resolve_destination_path(
                            filename, event_folder, file_index
                        )

                        if dest_path:
                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                            shutil.copy2(source_path, dest_path)
                            downloaded_files.append(dest_path)
                            self._mark_file_as_found(dest_path, path_lookup)

                            if matched_entry:
                                self.logger.info(
                                    f"Copied config file: {source_path} -> {matched_entry['relative_path']}"
                                )
                            else:
                                self.logger.info(
                                    f"Copied unmapped file: {source_path} -> {dest_path}"
                                )
                        else:
                            self.logger.warning(f"No mapping found for file: {filename}")

            elif uri_or_path.startswith('azureml://'):
                self.logger.info(f"Input is an Azure ML URI: {uri_or_path}")
                fs = AzureMachineLearningFileSystem(uri_or_path)
                for file_path in fs.ls():
                    filename = os.path.basename(file_path)
                    dest_path, matched_entry = self._resolve_destination_path(
                        filename, event_folder, file_index
                    )

                    if dest_path:
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        with fs.open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
                            dst.write(src.read())
                        downloaded_files.append(dest_path)
                        self._mark_file_as_found(dest_path, path_lookup)

                        if matched_entry:
                            self.logger.info(
                                f"Downloaded config file: {file_path} -> {matched_entry['relative_path']}"
                            )
                        else:
                            self.logger.info(
                                f"Downloaded unmapped file: {file_path} -> {dest_path}"
                            )
                    else:
                        self.logger.warning(f"No mapping found for file: {filename}")
            else:
                raise ValueError(f"Input path format not recognized: {uri_or_path}")
        except Exception as e:
            self.logger.error(f"Error processing input data: {str(e)}")
            raise

        self._report_missing_config_files(file_index)
        self.logger.info(f"Successfully processed {len(downloaded_files)} files")
        return downloaded_files

    def _extract_file_paths(self, value: Any) -> List[str]:
        """Recursively collect string paths from nested config structures."""
        paths: List[str] = []
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                paths.append(cleaned)
        elif isinstance(value, dict):
            for nested_value in value.values():
                paths.extend(self._extract_file_paths(nested_value))
        elif isinstance(value, list):
            for item in value:
                paths.extend(self._extract_file_paths(item))
        return paths

    def _normalize_config_path(self, path_value: str) -> str:
        """Normalize relative config paths to use forward slashes and no leading ./"""
        normalized = path_value.strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized.lstrip("/")

    def _resolve_existing_path(self, path_value: str) -> Optional[str]:
        """Resolve a config-declared path to an existing absolute path if possible."""
        if not path_value:
            return None

        candidates: List[str] = []
        cleaned = str(path_value).strip()

        if os.path.isabs(cleaned):
            candidates.append(os.path.normpath(cleaned))
        else:
            normalized_rel = self._normalize_config_path(cleaned)
            candidates.append(os.path.normpath(os.path.join(root_dir, normalized_rel)))
            candidates.append(os.path.normpath(os.path.join(project_root, normalized_rel)))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate

        return None

    def _build_config_file_index(self) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
        """Create lookup tables for config-defined input files."""
        sections: Dict[str, Any] = {
            'input_files': self.config.get('input_files', {}),
            'scan_files': self.config.get('scan_files', {}),
            'session_files': self.config.get('session_files', {})
        }

        post_analysis_config = self.config.get('post_analysis_mode') or {}
        for pa_section in ('scan_files', 'entry_scan_files'):
            if pa_section in post_analysis_config:
                sections[f'post_analysis_mode.{pa_section}'] = post_analysis_config.get(pa_section)

        neo4j_cfg = self.config.get('neo4j', {}) or {}
        additional_sources = {
            'neo4j.job_stream_mapping.file': (neo4j_cfg.get('job_stream_mapping', {}) or {}).get('file'),
            'neo4j.specialization_stream_mapping.file': (neo4j_cfg.get('specialization_stream_mapping', {}) or {}).get('file'),
        }

        for section_name, value in additional_sources.items():
            if value:
                sections[section_name] = value

        name_index: Dict[str, List[Dict[str, Any]]] = {}
        path_lookup: Dict[str, List[Dict[str, Any]]] = {}

        for section_name, section_value in sections.items():
            if not section_value:
                continue

            for raw_path in self._extract_file_paths(section_value):
                cleaned = raw_path.strip()
                if not cleaned:
                    continue

                if os.path.isabs(cleaned):
                    destination_path = os.path.normpath(cleaned)
                    display_path = cleaned
                else:
                    normalized_rel = self._normalize_config_path(cleaned)
                    destination_path = os.path.normpath(os.path.join(root_dir, normalized_rel))
                    display_path = normalized_rel

                filename = os.path.basename(destination_path).lower()
                if not filename:
                    continue

                entry = {
                    'relative_path': display_path,
                    'absolute_path': destination_path,
                    'section': section_name,
                    'found': False
                }

                name_index.setdefault(filename, []).append(entry)
                normalized_abs = os.path.normcase(destination_path)
                path_lookup.setdefault(normalized_abs, []).append(entry)

        return name_index, path_lookup

    def _match_config_entry(self, filename: str, file_index: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Find the config entry that matches the provided filename."""
        key = filename.lower()
        if key in file_index:
            for entry in file_index[key]:
                if not entry['found']:
                    return entry
            return file_index[key][0]

        for candidate, entries in file_index.items():
            if candidate in key or key in candidate:
                for entry in entries:
                    if not entry['found']:
                        return entry
                return entries[0]

        return None

    def _resolve_destination_path(
        self,
        filename: str,
        event_folder: str,
        file_index: Dict[str, List[Dict[str, Any]]]
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Determine where an input file should be stored for processing."""
        entry = self._match_config_entry(filename, file_index)
        if entry:
            return entry['absolute_path'], entry

        fallback_path = os.path.join(event_folder, filename)
        return fallback_path, None

    def _mark_file_as_found(
        self,
        destination_path: str,
        path_lookup: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Mark config entries as satisfied for the provided destination."""
        normalized = os.path.normcase(os.path.abspath(destination_path))
        if normalized in path_lookup:
            for entry in path_lookup[normalized]:
                entry['found'] = True

    def _report_missing_config_files(self, file_index: Dict[str, List[Dict[str, Any]]]) -> None:
        """Log any config-declared inputs that were not found in the payload."""
        if not file_index:
            return

        missing: List[Dict[str, Any]] = []
        for entries in file_index.values():
            for entry in entries:
                if not entry['found']:
                    missing.append(entry)

        if missing:
            self.logger.warning("Config declared input files missing from payload:")
            for entry in missing:
                self.logger.warning(
                    f"  • {entry['relative_path']} (section: {entry['section']})"
                )
        else:
            self.logger.info("All config-declared input files were located in the payload.")

    def _resolve_output_directory(self) -> str:
        """Return the absolute output directory as defined in config or defaults."""
        configured = self.config.get('output_dir')
        if configured:
            normalized = configured if os.path.isabs(configured) else os.path.join(
                root_dir, self._normalize_config_path(configured)
            )
            return os.path.normpath(normalized)

        event_name = self.config.get('event', {}).get('name', 'ecomm')
        return os.path.join(root_dir, 'data', event_name)

    def _append_output_file(
        self,
        collection: List[Tuple[str, str]],
        base_dir: str,
        filename: Optional[str],
        dest_name: Optional[str] = None
    ) -> None:
        """Helper to add config-declared output files to copy list."""
        if not filename:
            return

        cleaned = str(filename).strip()
        if not cleaned:
            return

        source_path = cleaned if os.path.isabs(cleaned) else os.path.join(base_dir, cleaned)
        normalized_source = os.path.normpath(source_path)
        destination = (dest_name or os.path.basename(cleaned) or cleaned).strip()
        if not destination:
            destination = os.path.basename(normalized_source)

        collection.append((normalized_source, destination))

    def _get_neo4j_support_files(self) -> List[Tuple[str, str]]:
        """Return absolute paths for Neo4j mapping files required downstream."""
        support_files: List[Tuple[str, str]] = []
        neo4j_cfg = self.config.get('neo4j', {})

        job_stream_file = (neo4j_cfg.get('job_stream_mapping', {}) or {}).get('file')
        specialization_file = (neo4j_cfg.get('specialization_stream_mapping', {}) or {}).get('file')

        for label, path_value in (
            ('job stream', job_stream_file),
            ('specialization', specialization_file),
        ):
            if not path_value:
                continue

            resolved = self._resolve_existing_path(path_value)
            if resolved:
                support_files.append((resolved, os.path.basename(resolved)))
            else:
                self.logger.warning(
                    "Neo4j %s mapping file declared but not found: %s",
                    label,
                    path_value,
                )

        return support_files

    def _build_output_file_mappings(self) -> Dict[str, List[Tuple[str, str]]]:
        """Construct Azure output mappings based on config-defined artifacts."""
        mappings: Dict[str, List[Tuple[str, str]]] = {
            'output_registration': [],
            'output_scan': [],
            'output_session': []
        }

        output_root = self._resolve_output_directory()
        standard_output_dir = os.path.join(output_root, 'output')

        output_files = self.config.get('output_files', {})
        session_bucket = mappings['output_session']

        def _append_support_file(path_value: Optional[str], context: str) -> None:
            """Ensure supplemental config files are copied alongside session outputs."""
            if not path_value:
                return

            resolved = self._resolve_existing_path(path_value)
            if resolved:
                dest_name = os.path.basename(resolved)
                session_bucket.append((resolved, dest_name))
                self.logger.info(
                    "Queued %s for session output transfer: %s", context, dest_name
                )
            else:
                self.logger.warning(
                    "Config-declared %s file not found prior to copy: %s",
                    context,
                    path_value,
                )

        combined_registration = output_files.get('combined_demographic_registration', {})
        for key in ('this_year', 'last_year_main', 'last_year_secondary'):
            self._append_output_file(
                mappings['output_registration'], standard_output_dir, combined_registration.get(key)
            )

        registration_with_demo = output_files.get('registration_with_demographic', {})
        for key in ('this_year', 'last_year_main', 'last_year_secondary'):
            self._append_output_file(
                mappings['output_registration'], standard_output_dir, registration_with_demo.get(key)
            )

        processed_demographics = output_files.get('processed_demographic_data', {})
        for key in ('this_year', 'last_year_main', 'last_year_secondary'):
            self._append_output_file(
                mappings['output_registration'], standard_output_dir, processed_demographics.get(key)
            )

        scan_outputs = self.config.get('scan_output_files', {})
        processed_scans = scan_outputs.get('processed_scans', {})
        for key in ('this_year', 'last_year_main', 'last_year_secondary', 'this_year_post'):
            self._append_output_file(
                mappings['output_scan'], standard_output_dir, processed_scans.get(key)
            )

        sessions_visited = scan_outputs.get('sessions_visited', {})
        for value in sessions_visited.values():
            self._append_output_file(mappings['output_scan'], standard_output_dir, value)

        attended_inputs = scan_outputs.get('attended_session_inputs', {})
        for value in attended_inputs.values():
            self._append_output_file(mappings['output_scan'], standard_output_dir, value)

        session_outputs = self.config.get('session_output_files', {})
        processed_sessions = session_outputs.get('processed_sessions', {})
        for key in ('this_year', 'last_year_main', 'last_year_secondary'):
            self._append_output_file(
                session_bucket, standard_output_dir, processed_sessions.get(key)
            )

        streams_catalog = session_outputs.get('streams_catalog')
        self._append_output_file(session_bucket, standard_output_dir, streams_catalog,
                                 dest_name=os.path.basename(streams_catalog) if streams_catalog else None)

        # Legacy fallback for cached streams to assist downstream processors
        self._append_output_file(session_bucket, standard_output_dir, 'streams_cache.json', dest_name='streams_cache.json')

        neo4j_support_files = self._get_neo4j_support_files()
        if neo4j_support_files:
            for abs_path, dest_name in neo4j_support_files:
                session_bucket.append((abs_path, dest_name))

        raw_session_inputs = self.config.get('session_files', {})
        for key, value in (raw_session_inputs or {}).items():
            _append_support_file(value, f"session_files.{key}")

        theatre_limits_cfg = (self.config.get('recommendation', {}) or {}).get('theatre_capacity_limits', {}) or {}
        if theatre_limits_cfg.get('enabled', True):
            for field_name in ('capacity_file', 'session_file'):
                _append_support_file(
                    theatre_limits_cfg.get(field_name),
                    f"recommendation.theatre_capacity_limits.{field_name}"
                )
        else:
            self.logger.info(
                "Theatre capacity enforcement disabled; skipping capacity/session support files"
            )

        # Remove empty mappings to avoid unnecessary copy attempts
        return {key: value for key, value in mappings.items() if value}
    
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
        
        original_cwd = os.getcwd()
        self.logger.info(f"Current working directory: {original_cwd}")
        
        os.chdir(root_dir)
        self.logger.info(f"Changed working directory to: {root_dir}")
        
        try:
            processor = RegistrationProcessor(self.config)

            vet_active = self._activate_vet_specific_functions(processor)
            if vet_active:
                self.logger.info("Veterinary-specific registration logic enabled for this run")
                practices_path = (self.config.get('input_files', {}) or {}).get('practices')
                if practices_path:
                    resolved_practices = self._resolve_existing_path(practices_path)
                    if resolved_practices:
                        self.logger.info(
                            "Practice-matching dataset detected for AML run: %s",
                            resolved_practices,
                        )
                    else:
                        self.logger.warning(
                            "Practice-matching dataset declared (%s) but not found before processing",
                            practices_path,
                        )
            
            if self.config.get('processors', {}).get('registration_processing', {}).get('enabled', True):
                processor.process()
                
                output_dir = self.config.get('output_dir', 'output')
                output_files: List[str] = []
                
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
                
                self.logger.info("Registration processing completed successfully")
                return result
            else:
                self.logger.info("Registration processing is disabled in config")
                return {'status': 'skipped', 'reason': 'disabled in config'}
                
        except Exception as e:
            self.logger.error(f"Registration processing failed: {str(e)}")
            raise
        finally:
            os.chdir(original_cwd)
            self.logger.info(f"Restored working directory to: {original_cwd}")
    
    def run_scan_processing(self) -> Dict[str, Any]:
        """
        Run scan processing using PA processor.
        
        Returns:
            Processing results and output paths
        """
        self.logger.info("Starting Scan Processing")
        
        original_cwd = os.getcwd()
        self.logger.info(f"Current working directory: {original_cwd}")
        
        os.chdir(root_dir)
        self.logger.info(f"Changed working directory to: {root_dir}")
        
        try:
            processor = ScanProcessor(self.config)
            
            if self.config.get('processors', {}).get('scan_processing', {}).get('enabled', True):
                processor.process()
                
                output_dir = self.config.get('output_dir', 'output')
                output_files: List[str] = []
                
                expected_files: List[str] = []
                scan_outputs = (self.config.get('scan_output_files', {}) or {})
                processed_scans = scan_outputs.get('processed_scans') or scan_outputs.get('scan_data') or {}
                sessions_visited = scan_outputs.get('sessions_visited') or {}

                for key in ('last_year_main', 'last_year_secondary', 'this_year', 'this_year_post'):
                    filename = processed_scans.get(key)
                    if filename:
                        expected_files.append(filename)

                for key in ('main_event', 'secondary_event', 'this_year_post'):
                    filename = sessions_visited.get(key)
                    if filename:
                        expected_files.append(filename)

                # Ensure we do not look for duplicates
                expected_files = list({name for name in expected_files if name})
                
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
                
                self.logger.info("Scan processing completed successfully")
                return result
            else:
                self.logger.info("Scan processing is disabled in config")
                return {'status': 'skipped', 'reason': 'disabled in config'}
                
        except Exception as e:
            self.logger.error(f"Scan processing failed: {str(e)}")
            raise
        finally:
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
        
        file_mappings = self._build_output_file_mappings()
        output_root = self._resolve_output_directory()
        
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
                    possible_sources = [
                        source_file,
                        os.path.join(output_root, dest_name),
                        os.path.join(output_root, os.path.basename(dest_name)),
                        os.path.join(root_dir, 'data', 'output', dest_name),
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
    configure_app_insights(service_name="pa_step1_data_prep")
    
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


def _bool_arg(value: Optional[str]) -> bool:
    """Convert an optional CLI value to boolean with support for flags."""
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
        nargs="?",
        const=True,
        default=False,
        type=_bool_arg,
        help="Run incremental processing (only new data). Accepts true/false."
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