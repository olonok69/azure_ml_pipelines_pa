"""
Simplified MLflow utilities for logging pipeline configuration and metrics.
Single run approach - no nested runs.
Enhanced to log all metrics from processing_summary.json for steps 4-10.
FIXED: Restored configuration file logging functionality from old version.
"""
import os
import logging
import mlflow
import yaml
import json
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MLflowManager:
    """Manages MLflow experiment tracking for the pipeline."""
    
    def __init__(self):
        """Initialize MLflow manager and verify environment setup."""
        self.experiment_id = None
        self.databricks_host = None
        self.mlflow_tracking_uri = None
        self.run_id = None
        self.run = None
        
    def verify_environment(self) -> bool:
        """
        Verify that all required MLflow environment variables are set.
        
        Returns:
            bool: True if all required environment variables are set, False otherwise
        
        Raises:
            Exception: If environment variables are not configured correctly
        """
        self.experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
        self.databricks_host = os.environ.get("DATABRICKS_HOST")
        self.mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        
        if self.experiment_id is None or self.databricks_host is None or self.mlflow_tracking_uri is None:
            error_msg = (
                "Environment variables are not configured correctly. "
                "Please set MLFLOW_EXPERIMENT_ID, DATABRICKS_HOST, and MLFLOW_TRACKING_URI"
            )
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        try:
            mlflow.create_experiment(self.experiment_id)
        except:
            logger.warning(f"Experiment {self.experiment_id} already existse.")
        mlflow.set_experiment(self.experiment_id)
        
        logger.info(f"MLflow environment verified successfully")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Databricks Host: {self.databricks_host}")
        logger.info(f"Tracking URI: {self.mlflow_tracking_uri}")
        
        return True
    
    def start_run(self, config: Dict[str, Any]) -> str:
        """
        Start a single MLflow run for the entire pipeline.
        FIXED: Restored configuration logging functionality from old version.
        
        Args:
            config: Pipeline configuration dictionary
            
        Returns:
            str: Run ID
        """
        # Generate run name
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        event_config = config.get("event", {})
        event_name = event_config.get("name", "pipeline")
        run_name = f"{event_name}_{timestamp}"
        
        logger.info(f"Starting MLflow run: {run_name}")
        
        # Start the run
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id
        
        # Log configuration as parameters (RESTORED from old version)
        logger.info("Logging configuration parameters to MLflow")
        self.log_config_as_params(config)
        
        # Log additional metadata
        mlflow.log_param("pipeline_type", "recommendation_pipeline")
        mlflow.log_param("event_name", event_name)
        mlflow.log_param("timestamp", timestamp)
        mlflow.log_param("pipeline_version", "enhanced")
        
        # Determine processing mode
        main_event_name = event_config.get("main_event_name", "").lower()
        processing_mode = "veterinary" if main_event_name in ["bva", "veterinary", "vet"] else "generic"
        mlflow.log_param("processing_mode", processing_mode)
        
        # Try to log the configuration file as an artifact (RESTORED from old version)
        try:
            safe_config = self._remove_sensitive_info(config)
            config_file = f"config_{event_name}_{timestamp}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(safe_config, f, default_flow_style=False)
            mlflow.log_artifact(config_file)
            os.remove(config_file)
            logger.info("Successfully logged configuration file as artifact")
        except Exception as e:
            logger.warning(f"Could not log configuration artifact: {e}")
        
        logger.info(f"Started MLflow run with ID: {self.run_id}")
        
        return self.run_id
    
    def log_config_as_params(self, config: Dict[str, Any], prefix: str = "") -> None:
        """
        Recursively log configuration parameters to MLflow, excluding sensitive information.
        RESTORED: This method was in the old version but missing in current version.
        
        Args:
            config: Configuration dictionary to log
            prefix: Prefix for parameter names (for nested configurations)
        """
        sensitive_keys = {"password", "secret", "key", "token", "credentials", "api_key"}
        
        for key, value in config.items():
            # Skip sensitive information
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                logger.debug(f"Skipping sensitive parameter: {key}")
                continue
            
            param_name = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # For nested dictionaries, recurse with updated prefix
                self.log_config_as_params(value, prefix=f"{param_name}.")
            elif isinstance(value, (list, tuple)):
                # For lists/tuples, convert to string representation
                if len(str(value)) <= 250:  # MLflow param value limit
                    try:
                        mlflow.log_param(param_name, str(value))
                        logger.debug(f"Logged parameter: {param_name} = {value}")
                    except Exception as e:
                        logger.warning(f"Could not log parameter {param_name}: {e}")
            else:
                # For scalar values, log directly
                if value is not None and len(str(value)) <= 250:
                    try:
                        mlflow.log_param(param_name, value)
                        logger.debug(f"Logged parameter: {param_name} = {value}")
                    except Exception as e:
                        logger.warning(f"Could not log parameter {param_name}: {e}")
    
    def log_config_params(self, config: Dict[str, Any]) -> None:
        """
        Log configuration parameters to MLflow.
        KEPT: This method is still used for backward compatibility.
        
        Args:
            config: Configuration dictionary
        """
        # Log key configuration parameters
        params_to_log = {
            "show_name": config.get("neo4j", {}).get("show_name", "unknown"),
            "config_file": config.get("config_file_path", "unknown"),
            "create_only_new": config.get("create_only_new", True),
            "input_source": config.get("input_source", "unknown"),
        }
        
        # Add pipeline steps configuration
        pipeline_steps = config.get("pipeline_steps", {})
        for step_name, enabled in pipeline_steps.items():
            params_to_log[f"step_{step_name}"] = enabled
        
        # Log each parameter
        for param_name, value in params_to_log.items():
            if isinstance(value, dict):
                # For nested dictionaries, convert to JSON string
                if len(str(value)) <= 250:  # MLflow param value limit
                    try:
                        mlflow.log_param(param_name, json.dumps(value))
                        logger.debug(f"Logged parameter: {param_name} = {value}")
                    except Exception as e:
                        logger.warning(f"Could not log parameter {param_name}: {e}")
            elif isinstance(value, (list, tuple)):
                # For lists/tuples, convert to string representation
                if len(str(value)) <= 250:  # MLflow param value limit
                    try:
                        mlflow.log_param(param_name, str(value))
                        logger.debug(f"Logged parameter: {param_name} = {value}")
                    except Exception as e:
                        logger.warning(f"Could not log parameter {param_name}: {e}")
            else:
                # For scalar values, log directly
                if value is not None and len(str(value)) <= 250:
                    try:
                        mlflow.log_param(param_name, value)
                        logger.debug(f"Logged parameter: {param_name} = {value}")
                    except Exception as e:
                        logger.warning(f"Could not log parameter {param_name}: {e}")
    
    def _remove_sensitive_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove sensitive information from configuration dictionary.
        RESTORED: This method was in the old version but missing in current version.
        
        Args:
            config: Original configuration dictionary
            
        Returns:
            Dict with sensitive information removed
        """
        import copy
        safe_config = copy.deepcopy(config)
        sensitive_keys = {"password", "secret", "key", "token", "credentials", "api_key"}
        
        def remove_sensitive(d):
            if isinstance(d, dict):
                keys_to_remove = []
                for k, v in d.items():
                    if any(sensitive in k.lower() for sensitive in sensitive_keys):
                        keys_to_remove.append(k)
                    elif isinstance(v, dict):
                        remove_sensitive(v)
                
                for k in keys_to_remove:
                    d[k] = "***REDACTED***"
        
        remove_sensitive(safe_config)
        return safe_config
    
    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            prefix: Optional prefix for metric names
        """
        for metric_name, metric_value in metrics.items():
            full_name = f"{prefix}_{metric_name}" if prefix else metric_name
            
            if isinstance(metric_value, (int, float)):
                try:
                    mlflow.log_metric(full_name, metric_value)
                    logger.debug(f"Logged metric: {full_name} = {metric_value}")
                except Exception as e:
                    logger.warning(f"Could not log metric {full_name}: {e}")
    
    def log_summary_metrics(self, processors: Dict[str, Any]) -> None:
        """
        Log summary metrics from all processors.
        Enhanced to log metrics from steps 4-10.
        
        Args:
            processors: Dictionary of processor instances
        """
        metrics = {}
        
        # Registration processor metrics - check for different attribute names
        if "reg_processor" in processors:
            reg = processors["reg_processor"]
            
            # Initial data loaded
            if hasattr(reg, 'df_bva'):
                metrics['registration_main_event_raw'] = len(reg.df_bva)
            if hasattr(reg, 'df_lvs'):
                metrics['registration_secondary_event_raw'] = len(reg.df_lvs)
            if hasattr(reg, 'df_bva_demo'):
                metrics['registration_demographic_main_raw'] = len(reg.df_bva_demo)
            if hasattr(reg, 'df_lvs_demo'):
                metrics['registration_demographic_secondary_raw'] = len(reg.df_lvs_demo)
            
            # Processed data - this year
            if hasattr(reg, 'df_bva_this_year'):
                metrics['registration_this_year_total'] = len(reg.df_bva_this_year)
                metrics['registration_unique_visitors_this_year'] = reg.df_bva_this_year['BadgeId'].nunique()
            elif hasattr(reg, 'df_bva_25_only_valid'):
                metrics['registration_this_year_valid'] = len(reg.df_bva_25_only_valid)
                metrics['registration_unique_visitors_this_year'] = reg.df_bva_25_only_valid['BadgeId'].nunique()
            
            # Processed data - last year
            if hasattr(reg, 'df_bva_last_year'):
                metrics['registration_last_year_main_total'] = len(reg.df_bva_last_year)
            elif hasattr(reg, 'df_bva_24_only_valid'):
                metrics['registration_last_year_main_valid'] = len(reg.df_bva_24_only_valid)
            
            # Returning visitors
            if hasattr(reg, 'df_bva_returning'):
                metrics['registration_returning_main'] = len(reg.df_bva_returning)
            elif hasattr(reg, 'df_bva_24_25_only_valid'):
                metrics['registration_returning_main_valid'] = len(reg.df_bva_24_25_only_valid)
            
            if hasattr(reg, 'df_lvs_returning'):
                metrics['registration_returning_secondary'] = len(reg.df_lvs_returning)
            elif hasattr(reg, 'df_lva_24_25_only_valid'):
                metrics['registration_returning_secondary_valid'] = len(reg.df_lva_24_25_only_valid)
            
            # Combined registration with demographics
            if hasattr(reg, 'df_reg_demo_this'):
                metrics['registration_with_demographics_this_year'] = len(reg.df_reg_demo_this)
            if hasattr(reg, 'df_reg_demo_last_bva'):
                metrics['registration_with_demographics_last_main'] = len(reg.df_reg_demo_last_bva)
            if hasattr(reg, 'df_reg_demo_last_lva'):
                metrics['registration_with_demographics_last_secondary'] = len(reg.df_reg_demo_last_lva)
        
        # Scan processor metrics - add storage for later use in dataframe
        if "scan_processor" in processors:
            scan = processors["scan_processor"]
            
            # Store dataframes for later inspection  
            if hasattr(scan, 'df_scan_last_bva'):
                metrics['scan_total_last_year_main'] = len(scan.df_scan_last_bva)
                metrics['scan_unique_seminars_last_year_main'] = scan.df_scan_last_bva['seminar_name'].nunique()
                metrics['scan_unique_attendees_last_year_main'] = scan.df_scan_last_bva['BadgeId'].nunique()
            if hasattr(scan, 'df_scan_last_lva'):
                metrics['scan_total_last_year_secondary'] = len(scan.df_scan_last_lva)
                metrics['scan_unique_seminars_last_year_secondary'] = scan.df_scan_last_lva['seminar_name'].nunique()
                metrics['scan_unique_attendees_last_year_secondary'] = scan.df_scan_last_lva['BadgeId'].nunique()
            
            # Check for processed scan data with different names
            if hasattr(scan, 'seminars_scans_past_enhanced_main'):
                metrics['scan_enhanced_main'] = len(scan.seminars_scans_past_enhanced_main)
            if hasattr(scan, 'seminars_scans_past_enhanced_secondary'):
                metrics['scan_enhanced_secondary'] = len(scan.seminars_scans_past_enhanced_secondary)
        
        # Session processor metrics
        if "session_processor" in processors:
            session = processors["session_processor"]
            
            # Check for different attribute names
            if hasattr(session, 'df_session_this'):
                metrics['session_total_this_year'] = len(session.df_session_this)
            elif hasattr(session, 'session_this_filtered_valid_cols'):
                metrics['session_total_this_year'] = len(session.session_this_filtered_valid_cols)
            
            if hasattr(session, 'df_session_last_bva'):
                metrics['session_total_last_year_main'] = len(session.df_session_last_bva)
            elif hasattr(session, 'session_last_filtered_valid_cols_bva'):
                metrics['session_total_last_year_main'] = len(session.session_last_filtered_valid_cols_bva)
            
            if hasattr(session, 'df_session_last_lva'):
                metrics['session_total_last_year_secondary'] = len(session.df_session_last_lva)
            elif hasattr(session, 'session_last_filtered_valid_cols_lva'):
                metrics['session_total_last_year_secondary'] = len(session.session_last_filtered_valid_cols_lva)
            
            if hasattr(session, 'streams'):
                metrics['session_unique_stream_categories'] = len(session.streams)
        
        # Neo4j Visitor processor metrics (Step 4)
        if "neo4j_visitor_processor" in processors:
            neo4j_visitor = processors["neo4j_visitor_processor"]
            if hasattr(neo4j_visitor, 'statistics'):
                stats = neo4j_visitor.statistics
                # Nodes created
                metrics['neo4j_visitor_nodes_created_this_year'] = stats['nodes_created'].get('visitor_this_year', 0)
                metrics['neo4j_visitor_nodes_created_last_year_bva'] = stats['nodes_created'].get('visitor_last_year_bva', 0)
                metrics['neo4j_visitor_nodes_created_last_year_lva'] = stats['nodes_created'].get('visitor_last_year_lva', 0)
                metrics['neo4j_visitor_total_nodes_created'] = sum(stats['nodes_created'].values())
                # Nodes skipped
                metrics['neo4j_visitor_nodes_skipped_this_year'] = stats['nodes_skipped'].get('visitor_this_year', 0)
                metrics['neo4j_visitor_nodes_skipped_last_year_bva'] = stats['nodes_skipped'].get('visitor_last_year_bva', 0)
                metrics['neo4j_visitor_nodes_skipped_last_year_lva'] = stats['nodes_skipped'].get('visitor_last_year_lva', 0)
                metrics['neo4j_visitor_total_nodes_skipped'] = sum(stats['nodes_skipped'].values())
        
        # Neo4j Session processor metrics (Step 5)
        if "neo4j_session_processor" in processors:
            neo4j_session = processors["neo4j_session_processor"]
            if hasattr(neo4j_session, 'statistics'):
                stats = neo4j_session.statistics
                # Nodes created
                metrics['neo4j_session_nodes_created_this_year'] = stats['nodes_created'].get('sessions_this_year', 0)
                metrics['neo4j_session_nodes_created_past_year_bva'] = stats['nodes_created'].get('sessions_past_year_bva', 0)
                metrics['neo4j_session_nodes_created_past_year_lva'] = stats['nodes_created'].get('sessions_past_year_lva', 0)
                metrics['neo4j_session_nodes_created_streams'] = stats['nodes_created'].get('streams', 0)
                metrics['neo4j_session_nodes_created_past_year'] = stats['nodes_created'].get('sessions_past_year', 0)
                metrics['neo4j_session_total_nodes_created'] = sum(stats['nodes_created'].values())
                # Nodes skipped
                metrics['neo4j_session_total_nodes_skipped'] = sum(stats['nodes_skipped'].values())
                # Relationships
                metrics['neo4j_session_relationships_created_this_year'] = stats['relationships_created'].get('sessions_this_year_has_stream', 0)
                metrics['neo4j_session_relationships_created_past_year'] = stats['relationships_created'].get('sessions_past_year_has_stream', 0)
                metrics['neo4j_session_total_relationships_created'] = sum(stats['relationships_created'].values())
                metrics['neo4j_session_total_relationships_skipped'] = sum(stats['relationships_skipped'].values())
        
        # Neo4j Job Stream processor metrics (Step 6)
        if "neo4j_job_stream_processor" in processors:
            neo4j_job_stream = processors["neo4j_job_stream_processor"]
            if hasattr(neo4j_job_stream, 'statistics'):
                stats = neo4j_job_stream.statistics
                if not stats.get('processing_skipped', False):
                    metrics['neo4j_job_stream_relationships_created'] = stats.get('relationships_created', 0)
                    metrics['neo4j_job_stream_relationships_skipped'] = stats.get('relationships_skipped', 0)
                    metrics['neo4j_job_stream_relationships_not_found'] = stats.get('relationships_not_found', 0)
                    metrics['neo4j_job_stream_stream_mappings_applied'] = stats.get('stream_mappings_applied', 0)
                    metrics['neo4j_job_stream_visitor_nodes_processed'] = stats.get('visitor_nodes_processed', 0)
                    metrics['neo4j_job_stream_job_roles_processed'] = stats.get('job_roles_processed', 0)
                    metrics['neo4j_job_stream_stream_matches_found'] = stats.get('stream_matches_found', 0)
                    metrics['neo4j_job_stream_initial_count'] = stats.get('initial_relationship_count', 0)
                    metrics['neo4j_job_stream_final_count'] = stats.get('final_relationship_count', 0)
        
        # Neo4j Specialization Stream processor metrics (Step 7)
        if "neo4j_specialization_stream_processor" in processors:
            neo4j_spec_stream = processors["neo4j_specialization_stream_processor"]
            if hasattr(neo4j_spec_stream, 'statistics'):
                stats = neo4j_spec_stream.statistics
                if not stats.get('processing_skipped', False):
                    metrics['neo4j_spec_stream_initial_count'] = stats.get('initial_count', 0)
                    metrics['neo4j_spec_stream_final_count'] = stats.get('final_count', 0)
                    metrics['neo4j_spec_stream_relationships_created'] = stats.get('relationships_created', 0)
                    metrics['neo4j_spec_stream_relationships_skipped'] = stats.get('relationships_skipped', 0)
                    metrics['neo4j_spec_stream_relationships_not_found'] = stats.get('relationships_not_found', 0)
                    metrics['neo4j_spec_stream_specializations_processed'] = stats.get('specializations_processed', 0)
                    metrics['neo4j_spec_stream_specializations_mapped'] = stats.get('specializations_mapped', 0)
                    metrics['neo4j_spec_stream_stream_matches_found'] = stats.get('stream_matches_found', 0)
                    metrics['neo4j_spec_stream_total_relationships_created'] = stats.get('total_relationships_created', 0)
                    metrics['neo4j_spec_stream_total_relationships_skipped'] = stats.get('total_relationships_skipped', 0)
                    # Visitor nodes processed
                    visitor_nodes = stats.get('visitor_nodes_processed', {})
                    if isinstance(visitor_nodes, dict):
                        metrics['neo4j_spec_stream_visitors_this_year'] = visitor_nodes.get('visitor_this_year', 0)
                        metrics['neo4j_spec_stream_visitors_last_year_bva'] = visitor_nodes.get('visitor_last_year_bva', 0)
                        metrics['neo4j_spec_stream_visitors_last_year_lva'] = visitor_nodes.get('visitor_last_year_lva', 0)
                        metrics['neo4j_spec_stream_total_visitors_processed'] = sum(visitor_nodes.values())
                    else:
                        metrics['neo4j_spec_stream_total_visitors_processed'] = visitor_nodes
        
        # Neo4j Visitor Relationship processor metrics (Step 8)
        if "neo4j_visitor_relationship_processor" in processors:
            neo4j_visitor_rel = processors["neo4j_visitor_relationship_processor"]
            if hasattr(neo4j_visitor_rel, 'statistics'):
                stats = neo4j_visitor_rel.statistics
                # Relationships created
                for rel_type, count in stats.get('relationships_created', {}).items():
                    metrics[f'neo4j_visitor_rel_created_{rel_type}'] = count
                metrics['neo4j_visitor_rel_total_created'] = sum(stats.get('relationships_created', {}).values())
                # Relationships skipped
                for rel_type, count in stats.get('relationships_skipped', {}).items():
                    metrics[f'neo4j_visitor_rel_skipped_{rel_type}'] = count
                metrics['neo4j_visitor_rel_total_skipped'] = sum(stats.get('relationships_skipped', {}).values())
                # Relationships failed
                for rel_type, count in stats.get('relationships_failed', {}).items():
                    metrics[f'neo4j_visitor_rel_failed_{rel_type}'] = count
                metrics['neo4j_visitor_rel_total_failed'] = sum(stats.get('relationships_failed', {}).values())
        
        # Session Embedding processor metrics (Step 9)
        if "session_embedding_processor" in processors:
            session_embedding = processors["session_embedding_processor"]
            if hasattr(session_embedding, 'statistics'):
                stats = session_embedding.statistics
                metrics['session_embedding_total_processed'] = stats.get('total_sessions_processed', 0)
                metrics['session_embedding_with_embeddings'] = stats.get('sessions_with_embeddings', 0)
                metrics['session_embedding_with_stream_descriptions'] = stats.get('sessions_with_stream_descriptions', 0)
                metrics['session_embedding_errors'] = stats.get('errors', 0)
                # Sessions by type
                sessions_by_type = stats.get('sessions_by_type', {})
                if sessions_by_type:
                    metrics['session_embedding_this_year'] = sessions_by_type.get('sessions_this_year', 0)
                    metrics['session_embedding_past_year'] = sessions_by_type.get('sessions_past_year', 0)
        
        # Session Recommendation processor metrics (Step 10)
        if "session_recommendation_processor" in processors:
            rec = processors["session_recommendation_processor"]
            if hasattr(rec, 'statistics'):
                stats = rec.statistics
                metrics['recommendations_visitors_processed'] = stats.get('visitors_processed', 0)
                metrics['recommendations_visitors_with_recommendations'] = stats.get('visitors_with_recommendations', 0)
                metrics['recommendations_visitors_without_recommendations'] = stats.get('visitors_without_recommendations', 0)
                metrics['recommendations_total_generated'] = stats.get('total_recommendations_generated', 0)
                metrics['recommendations_total_filtered'] = stats.get('total_filtered_recommendations', 0)
                metrics['recommendations_errors'] = stats.get('errors', 0)
                metrics['recommendations_processing_time'] = stats.get('processing_time', 0)
            # Legacy attributes for backward compatibility
            elif hasattr(rec, 'total_visitors_processed'):
                metrics['recommendations_visitors_processed'] = rec.total_visitors_processed
            if hasattr(rec, 'total_recommendations_created'):
                metrics['recommendations_total_created'] = rec.total_recommendations_created
        
        # Log all metrics
        self.log_metrics(metrics)
        logger.info(f"Logged {len(metrics)} summary metrics to MLflow")
    
    def log_processing_summary(self, summary_file_path: str) -> None:
        """
        Log the processing summary JSON as an artifact and extract metrics.
        Enhanced to extract all metrics from the JSON summary.
        
        Args:
            summary_file_path: Path to the summary JSON file
        """
        try:
            # Log the JSON file as an artifact
            if os.path.exists(summary_file_path):
                mlflow.log_artifact(summary_file_path)
                logger.info(f"Logged summary file as artifact: {summary_file_path}")
                
                # Also extract and log key metrics from the summary
                with open(summary_file_path, 'r') as f:
                    summary = json.load(f)
                
                # Log processing time
                if 'processing_time_seconds' in summary:
                    mlflow.log_metric('total_processing_time_seconds', summary['processing_time_seconds'])
                
                # Log step counts
                if 'steps_completed' in summary:
                    mlflow.log_metric('steps_completed', len(summary['steps_completed']))
                
                # Extract metrics from each section of the summary
                # This ensures we capture any metrics that might have been missed
                self._log_summary_section_metrics(summary)
                
        except Exception as e:
            logger.warning(f"Could not log summary file: {e}")
    
    def _log_summary_section_metrics(self, summary: Dict[str, Any]) -> None:
        """
        Helper method to extract and log metrics from summary JSON sections.
        
        Args:
            summary: Processing summary dictionary
        """
        try:
            # Helper function to flatten nested dictionaries
            def flatten_dict(d, parent_key='', sep='_'):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    elif isinstance(v, (int, float)):
                        items.append((new_key, v))
                return dict(items)
            
            # Process each section and log numeric metrics
            sections_to_process = [
                'neo4j_visitor', 'neo4j_session', 'neo4j_job_stream',
                'neo4j_specialization_stream', 'neo4j_visitor_relationship',
                'session_embedding', 'session_recommendation'
            ]
            
            for section in sections_to_process:
                if section in summary:
                    flattened = flatten_dict(summary[section], parent_key=f'summary_{section}')
                    for metric_name, metric_value in flattened.items():
                        if isinstance(metric_value, (int, float)) and not metric_name.endswith('_skipped'):
                            try:
                                mlflow.log_metric(metric_name, metric_value)
                                logger.debug(f"Logged summary metric: {metric_name} = {metric_value}")
                            except Exception as e:
                                logger.debug(f"Could not log summary metric {metric_name}: {e}")
        
        except Exception as e:
            logger.warning(f"Error logging summary section metrics: {e}")
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the MLflow run.
        
        Args:
            status: Status of the run (FINISHED, FAILED, etc.)
        """
        try:
            mlflow.end_run(status=status)
            logger.info(f"MLflow run ended with status: {status}")
        except Exception as e:
            logger.warning(f"Could not end MLflow run: {e}")