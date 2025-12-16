"""
Simplified main.py with MLflow integration using a single run.
Logs all summary metrics directly without nested runs.
Enhanced to capture all metrics from steps 4-10.
FIXED: Added config file path logging to MLflow.
"""
import argparse
import os
import sys
import logging
from datetime import datetime
import mlflow
from utils.logging_utils import setup_logging
from utils.config_utils import load_config
from utils.summary_utils import generate_and_save_summary
from utils.mlflow_utils import MLflowManager
from pipeline import (
    run_registration_processing,
    run_scan_processing,
    run_session_processing,
    run_neo4j_processing,
    run_session_recommendation_processing,
)


from dotenv import load_dotenv

def configure_azure_logging():
    """Configure Azure SDK logging to reduce verbosity."""
    azure_loggers = [
        'azure.core.pipeline.policies.http_logging_policy',
        'azure.storage',
        'azure.identity',
        'azure.core',
        'azure.storage.blob',
        'azure.storage.filedatalake',
    ]
    
    for logger_name in azure_loggers:
        azure_logger = logging.getLogger(logger_name)
        azure_logger.setLevel(logging.WARNING)  # Only show warnings and errors


def log_documentation_assets(config, logger, mlflow_manager=None):
    """Log documentation metadata and upload SVG assets when available."""
    doc_cfg = config.get("documentation")
    if not doc_cfg:
        logger.info("No documentation assets declared in configuration")
        return

    logged_artifacts = set()

    for label, doc_path in doc_cfg.items():
        if not doc_path:
            continue

        resolved_path = os.path.abspath(doc_path)
        exists = os.path.exists(resolved_path)
        status = "found" if exists else "missing"
        logger.info(
            "Documentation asset '%s': %s (%s)",
            label,
            doc_path,
            status,
        )

        if mlflow_manager:
            mlflow_manager.log_param(f"documentation_{label}", doc_path)
            if exists and resolved_path not in logged_artifacts:
                try:
                    mlflow.log_artifact(resolved_path)
                    logged_artifacts.add(resolved_path)
                    logger.info("Uploaded documentation artifact for '%s'", label)
                except Exception as artifact_error:  # pragma: no cover - defensive
                    logger.warning(
                        "Could not upload documentation asset '%s': %s",
                        label,
                        artifact_error,
                    )


def log_neo4j_step_metrics(mlflow_manager, processors, step_number):
    """
    Log metrics for specific Neo4j processing steps.
    
    Args:
        mlflow_manager: MLflow manager instance
        processors: Dictionary of processors
        step_number: The step number to log metrics for
    """
    if not mlflow_manager:
        return
    
    try:
        step_metrics = {}
        
        # Map step numbers to processor names and metric prefixes
        step_mapping = {
            4: ('neo4j_visitor_processor', 'step4_neo4j_visitor'),
            5: ('neo4j_session_processor', 'step5_neo4j_session'),
            6: ('neo4j_job_stream_processor', 'step6_neo4j_job_stream'),
            7: ('neo4j_specialization_stream_processor', 'step7_neo4j_spec_stream'),
            8: ('neo4j_visitor_relationship_processor', 'step8_neo4j_visitor_rel'),
            9: ('session_embedding_processor', 'step9_session_embedding'),
            10: ('session_recommendation_processor', 'step10_recommendations')
        }
        
        if step_number in step_mapping:
            processor_name, metric_prefix = step_mapping[step_number]
            
            if processor_name in processors:
                processor = processors[processor_name]
                
                if hasattr(processor, 'statistics'):
                    stats = processor.statistics
                    
                    # Log step-specific metrics based on the processor type
                    if step_number == 4:  # Neo4j Visitor
                        if 'nodes_created' in stats:
                            step_metrics[f'{metric_prefix}_nodes_created'] = sum(stats['nodes_created'].values())
                        if 'nodes_skipped' in stats:
                            step_metrics[f'{metric_prefix}_nodes_skipped'] = sum(stats['nodes_skipped'].values())
                        if 'nodes_updated' in stats:
                            step_metrics[f'{metric_prefix}_nodes_updated'] = sum(stats['nodes_updated'].values())
                    
                    elif step_number == 5:  # Neo4j Session
                        if 'nodes_created' in stats:
                            step_metrics[f'{metric_prefix}_nodes_created'] = sum(stats['nodes_created'].values())
                        if 'nodes_skipped' in stats:
                            step_metrics[f'{metric_prefix}_nodes_skipped'] = sum(stats['nodes_skipped'].values())
                        if 'relationships_created' in stats:
                            step_metrics[f'{metric_prefix}_relationships_created'] = sum(stats['relationships_created'].values())
                    
                    elif step_number in [6, 7]:  # Job Stream and Specialization Stream
                        if not stats.get('processing_skipped', False):
                            step_metrics[f'{metric_prefix}_relationships_created'] = stats.get('relationships_created', 0)
                            step_metrics[f'{metric_prefix}_relationships_skipped'] = stats.get('relationships_skipped', 0)
                    
                    elif step_number == 8:  # Visitor Relationships
                        if 'relationships_created' in stats:
                            step_metrics[f'{metric_prefix}_relationships_created'] = sum(stats['relationships_created'].values())
                            assisted_value = stats['relationships_created'].get('assisted_session_this_year')
                            if assisted_value is not None:
                                step_metrics[f'{metric_prefix}_assisted_session_this_year_created'] = assisted_value
                        if 'relationships_failed' in stats:
                            step_metrics[f'{metric_prefix}_relationships_failed'] = sum(stats['relationships_failed'].values())
                    
                    elif step_number == 9:  # Session Embedding
                        step_metrics[f'{metric_prefix}_sessions_processed'] = stats.get('total_sessions_processed', 0)
                        step_metrics[f'{metric_prefix}_sessions_with_embeddings'] = stats.get('sessions_with_embeddings', 0)
                    
                    elif step_number == 10:  # Recommendations
                        step_metrics[f'{metric_prefix}_visitors_processed'] = stats.get('visitors_processed', 0)
                        step_metrics[f'{metric_prefix}_recommendations_generated'] = stats.get('total_recommendations_generated', 0)
        
        # Log the metrics
        if step_metrics:
            mlflow_manager.log_metrics(step_metrics)
            logging.getLogger(__name__).debug(f"Logged {len(step_metrics)} metrics for step {step_number}")
    
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not log metrics for step {step_number}: {e}")


def main():
    """Main entry point for the pipeline with simplified MLflow integration."""
    
    load_dotenv("keys/.env")
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process and upload data to Neo4j with MLflow tracking.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--recreate-all",
        action="store_true",
        help="Recreate all nodes, even if they already exist in Neo4j",
    )
    parser.add_argument(
        "--skip-neo4j", 
        action="store_true", 
        help="Skip Neo4j upload steps"
    )
    parser.add_argument(
        "--only-steps",
        type=str,
        help="Comma-separated list of step numbers to run (e.g., '1,3,4')",
    )
    parser.add_argument(
        "--only-recommendations",
        action="store_true",
        help="Only run session recommendation processing (step 10)",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Skip MLflow tracking (useful for testing)",
    )
    args = parser.parse_args()

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)

    # Setup logging
    logger = setup_logging(log_file="logs/data_processing.log")
    configure_azure_logging()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # ADDED: Store config file path for MLflow logging
    config['config_file_path'] = args.config
    
    # Track pipeline start time
    pipeline_start_time = datetime.now()
    
    # Initialize MLflow manager if not skipped
    mlflow_manager = None
    run_id = None
    if not args.skip_mlflow:
        try:
            mlflow_manager = MLflowManager()
            mlflow_manager.verify_environment()
            run_id = mlflow_manager.start_run(config)
            logger.info(f"Started MLflow run: {run_id}")
            
            # ADDED: Log command line arguments
            mlflow_manager.log_param("command_line_args", str(vars(args)))
            mlflow_manager.log_param("config_file", args.config)
            
        except Exception as e:
            logger.warning(f"Could not initialize MLflow: {e}. Continuing without MLflow tracking.")
            logger.warning("To enable MLflow, set: MLFLOW_EXPERIMENT_ID, DATABRICKS_HOST, MLFLOW_TRACKING_URI")
            args.skip_mlflow = True
            mlflow_manager = None
    else:
        logger.info("MLflow tracking is disabled (--skip-mlflow flag)")

    # Surface documentation assets via logs and MLflow params/artifacts
    log_documentation_assets(config, logger, mlflow_manager)

    # Get create_only_new from config, but allow command line to override
    if args.recreate_all:
        create_only_new = False
    else:
        create_only_new = config.get("create_only_new", True)

    logger.info(f"Running with create_only_new={create_only_new}")
    
    # Log initial pipeline parameters to MLflow
    if mlflow_manager:
        mlflow_manager.log_param("create_only_new", create_only_new)
        mlflow_manager.log_param("skip_neo4j", args.skip_neo4j)

    try:
        # Determine which steps to run
        steps_to_run = []
        if args.only_steps:
            steps_to_run = [int(step.strip()) for step in args.only_steps.split(",")]
            if mlflow_manager:
                mlflow_manager.log_param("steps_to_run", str(steps_to_run))
            logger.info(f"User-specified steps_to_run: {steps_to_run}")

        # If only recommendations flag is set, only run step 10
        if args.only_recommendations:
            logger.info("Running only session recommendation processing (step 10)")
            
            processors = {
                "session_recommendation_processor": run_session_recommendation_processing(
                    config, create_only_new
                )
            }
            
            # Log metrics for recommendation processing
            if mlflow_manager:
                log_neo4j_step_metrics(mlflow_manager, processors, 10)
                mlflow_manager.log_summary_metrics(processors)
                mlflow_manager.log_param("steps_run", "10")
            
            generate_and_save_summary(processors, skip_neo4j=True)
            logger.info("Session recommendation processing completed successfully")
            
            # End MLflow run
            if mlflow_manager:
                processing_time = (datetime.now() - pipeline_start_time).total_seconds()
                mlflow_manager.log_metric("total_processing_time_seconds", processing_time)
                mlflow_manager.end_run(status="FINISHED")
                logger.info(f"MLflow run completed. Run ID: {run_id}")
            
            return

        # Run pipeline steps as configured
        processors = {}
        steps_completed = []

        # Step 1: Registration data processing
        if not steps_to_run or 1 in steps_to_run:
            if config.get("pipeline_steps", {}).get("registration_processing", True):
                logger.info("Starting step 1: Registration data processing")
                step_start = datetime.now()
                
                processors["reg_processor"] = run_registration_processing(config)
                
                # Log step timing
                if mlflow_manager:
                    step_time = (datetime.now() - step_start).total_seconds()
                    mlflow_manager.log_metric("step1_registration_time_seconds", step_time)
                
                steps_completed.append(1)
                logger.info("Completed step 1: Registration data processing")
            else:
                logger.info("Skipping step 1: Registration data processing (disabled in config)")

        # Step 2: Scan data processing
        if not steps_to_run or 2 in steps_to_run:
            if config.get("pipeline_steps", {}).get("scan_processing", True):
                logger.info("Starting step 2: Scan data processing")
                step_start = datetime.now()
                
                processors["scan_processor"] = run_scan_processing(config)
                
                # Log step timing
                if mlflow_manager:
                    step_time = (datetime.now() - step_start).total_seconds()
                    mlflow_manager.log_metric("step2_scan_time_seconds", step_time)
                
                steps_completed.append(2)
                logger.info("Completed step 2: Scan data processing")
            else:
                logger.info("Skipping step 2: Scan data processing (disabled in config)")

        # Step 3: Session data processing
        if not steps_to_run or 3 in steps_to_run:
            if config.get("pipeline_steps", {}).get("session_processing", True):
                logger.info("Starting step 3: Session data processing")
                step_start = datetime.now()
                
                processors["session_processor"] = run_session_processing(config)
                
                # Log step timing
                if mlflow_manager:
                    step_time = (datetime.now() - step_start).total_seconds()
                    mlflow_manager.log_metric("step3_session_time_seconds", step_time)
                
                steps_completed.append(3)
                logger.info("Completed step 3: Session data processing")
            else:
                logger.info("Skipping step 3: Session data processing (disabled in config)")

        # Neo4j processing steps (optional with --skip-neo4j flag)
        if not args.skip_neo4j:
            # Check which specific Neo4j steps should run
            neo4j_steps_to_run = []
            if steps_to_run:
                # Filter to only Neo4j steps (4-10)
                neo4j_steps_to_run = [s for s in steps_to_run if s >= 4 and s <= 10]
            else:
                # If no specific steps are requested, include all steps that are enabled in config
                if config.get("pipeline_steps", {}).get("neo4j_visitor_processing", True):
                    neo4j_steps_to_run.append(4)
                if config.get("pipeline_steps", {}).get("neo4j_session_processing", True):
                    neo4j_steps_to_run.append(5)
                if config.get("pipeline_steps", {}).get("neo4j_job_stream_processing", True):
                    neo4j_steps_to_run.append(6)
                if config.get("pipeline_steps", {}).get("neo4j_specialization_stream_processing", True):
                    neo4j_steps_to_run.append(7)
                if config.get("pipeline_steps", {}).get("neo4j_visitor_relationship_processing", True):
                    neo4j_steps_to_run.append(8)
                if config.get("pipeline_steps", {}).get("session_embedding_processing", True):
                    neo4j_steps_to_run.append(9)
                if config.get("pipeline_steps", {}).get("session_recommendation_processing", True):
                    neo4j_steps_to_run.append(10)

            # Pre-execution diagnostic matrix of Neo4j steps
            neo4j_step_names = {
                4: "Neo4j Visitor",
                5: "Neo4j Session",
                6: "Neo4j Job->Stream",
                7: "Neo4j Specialization->Stream",
                8: "Neo4j Visitor Relationships",
                9: "Session Embeddings",
                10: "Session Recommendations"
            }
            logger.info("Neo4j step selection summary:")
            for s in range(4, 11):
                enabled = config.get("pipeline_steps", {}).get({
                    4:"neo4j_visitor_processing",
                    5:"neo4j_session_processing",
                    6:"neo4j_job_stream_processing",
                    7:"neo4j_specialization_stream_processing",
                    8:"neo4j_visitor_relationship_processing",
                    9:"session_embedding_processing",
                    10:"session_recommendation_processing"
                }[s], True)
                selected = s in neo4j_steps_to_run
                status = ("RUN" if (enabled and selected) else
                          "DISABLED" if not enabled else
                          "NOT_SELECTED")
                logger.info(f"  Step {s}: {neo4j_step_names[s]:32} -> {status}")
            if create_only_new:
                logger.info("Neo4j session processing will run in INCREMENTAL mode (create_only_new=True)")
            else:
                logger.info("Neo4j session processing will run in FULL REBUILD mode (create_only_new=False)")

            if neo4j_steps_to_run:
                logger.info("Starting Neo4j data processing")
                neo4j_start = datetime.now()
                
                # Process each Neo4j step individually and log metrics after each
                for step_num in sorted(neo4j_steps_to_run):
                    step_start = datetime.now()
                    neo4j_processors = run_neo4j_processing(
                        config, create_only_new, [step_num]
                    )
                    processors.update(neo4j_processors)
                    
                    # Log metrics for this specific step
                    if mlflow_manager:
                        step_time = (datetime.now() - step_start).total_seconds()
                        mlflow_manager.log_metric(f"step{step_num}_time_seconds", step_time)
                        log_neo4j_step_metrics(mlflow_manager, processors, step_num)
                    
                    steps_completed.append(step_num)
                
                # Log Neo4j processing time
                if mlflow_manager:
                    neo4j_time = (datetime.now() - neo4j_start).total_seconds()
                    mlflow_manager.log_metric("neo4j_total_time_seconds", neo4j_time)
                
                logger.info("Completed Neo4j data processing")
            else:
                logger.info("Skipping Neo4j processing (all Neo4j steps disabled in config or not in selected steps)")

        # Generate and save summary statistics
        logger.info("Generating summary statistics")
        summary = generate_and_save_summary(processors, args.skip_neo4j)
        
        # Print key metrics to console for visibility
        logger.info("=" * 60)
        logger.info("KEY PIPELINE METRICS")
        logger.info("=" * 60)
        
        # Registration metrics
        if "reg_processor" in processors:
            reg = processors["reg_processor"]
            logger.info("Registration Metrics:")
            if hasattr(reg, 'df_bva'):
                logger.info(f"  Total main event registrations: {len(reg.df_bva)}")
            if hasattr(reg, 'df_lvs'):
                logger.info(f"  Total secondary event registrations: {len(reg.df_lvs)}")
            if hasattr(reg, 'df_bva_25_only_valid') or hasattr(reg, 'df_bva_this_year'):
                df = getattr(reg, 'df_bva_25_only_valid', getattr(reg, 'df_bva_this_year', None))
                if df is not None:
                    logger.info(f"  This year unique visitors: {df['BadgeId'].nunique()}")
        
        # Scan metrics
        if "scan_processor" in processors:
            scan = processors["scan_processor"]
            logger.info("Scan Metrics:")
            if hasattr(scan, 'df_scan_last_bva'):
                logger.info(f"  Last year main event scans: {len(scan.df_scan_last_bva)}")
                logger.info(f"  Unique attendees: {scan.df_scan_last_bva['BadgeId'].nunique()}")
            if hasattr(scan, 'df_scan_last_lva'):
                logger.info(f"  Last year secondary event scans: {len(scan.df_scan_last_lva)}")
        
        # Session metrics
        if "session_processor" in processors:
            session = processors["session_processor"]
            logger.info("Session Metrics:")
            if hasattr(session, 'df_session_this') or hasattr(session, 'session_this_filtered_valid_cols'):
                df = getattr(session, 'df_session_this', getattr(session, 'session_this_filtered_valid_cols', None))
                if df is not None:
                    logger.info(f"  This year sessions: {len(df)}")
            if hasattr(session, 'streams'):
                logger.info(f"  Unique stream categories: {len(session.streams)}")
        
        # Neo4j metrics summary
        for step_num in range(4, 11):
            step_names = {
                4: "Neo4j Visitor",
                5: "Neo4j Session",
                6: "Neo4j Job Stream",
                7: "Neo4j Specialization Stream",
                8: "Neo4j Visitor Relationship",
                9: "Session Embedding",
                10: "Session Recommendation"
            }
            processor_names = {
                4: "neo4j_visitor_processor",
                5: "neo4j_session_processor",
                6: "neo4j_job_stream_processor",
                7: "neo4j_specialization_stream_processor",
                8: "neo4j_visitor_relationship_processor",
                9: "session_embedding_processor",
                10: "session_recommendation_processor"
            }
            
            if processor_names[step_num] in processors:
                processor = processors[processor_names[step_num]]
                if hasattr(processor, 'statistics'):
                    stats = processor.statistics
                    logger.info(f"{step_names[step_num]} Metrics:")
                    
                    if step_num in [4, 5] and 'nodes_created' in stats:
                        logger.info(f"  Nodes created: {sum(stats['nodes_created'].values())}")
                    if step_num in [5, 6, 7, 8] and 'relationships_created' in stats:
                        if isinstance(stats['relationships_created'], dict):
                            logger.info(f"  Relationships created: {sum(stats['relationships_created'].values())}")
                        else:
                            logger.info(f"  Relationships created: {stats['relationships_created']}")
                    if step_num == 9:
                        logger.info(f"  Sessions processed: {stats.get('total_sessions_processed', 0)}")
                    if step_num == 10:
                        logger.info(f"  Visitors processed: {stats.get('visitors_processed', 0)}")
                        logger.info(f"  Recommendations generated: {stats.get('total_recommendations_generated', 0)}")
        
        logger.info("=" * 60)
        
        # Log all summary metrics to MLflow
        if mlflow_manager:
            # Log which processors were run
            mlflow_manager.log_param("processors_run", ", ".join(processors.keys()))
            mlflow_manager.log_param("steps_completed", str(steps_completed))
            
            # Log comprehensive summary metrics from all processors
            mlflow_manager.log_summary_metrics(processors)
            
            # Log total processing time
            total_time = (datetime.now() - pipeline_start_time).total_seconds()
            mlflow_manager.log_metric("total_processing_time_seconds", total_time)
            
            # Try to log summary file and extract additional metrics
            summary_json_path = "logs/processing_summary.json"
            if os.path.exists(summary_json_path):
                mlflow_manager.log_processing_summary(summary_json_path)
            
            # Try to log output artifacts
            try:
                # Log key output files if they exist
                output_dir = config.get("output_dir", "output")
                important_outputs = [
                    "output/Registration_data_bva_25_only_valid.csv",
                    "output/scan_last_filtered_valid_cols_bva.csv",
                    "output/session_this_filtered_valid_cols.csv",
                    "logs/processing_summary.json"
                ]
                
                for output_file in important_outputs:
                    file_path = os.path.join(output_dir, output_file) if not output_file.startswith("logs") else output_file
                    if os.path.exists(file_path):
                        mlflow.log_artifact(file_path)
                        logger.info(f"Logged artifact: {os.path.basename(file_path)}")
            except Exception as e:
                logger.warning(f"Could not log some artifacts: {e}")
            
            # End the run
            mlflow_manager.end_run(status="FINISHED")
            logger.info(f"MLflow run completed successfully. Run ID: {run_id}")
            logger.info(f"View run at: {mlflow_manager.mlflow_tracking_uri}/#/experiments/{mlflow_manager.experiment_id}/runs/{run_id}")
        
        logger.info("Data processing and analysis completed successfully")

    except Exception as e:
        logger.error(f"Error in data processing: {e}", exc_info=True)
        
        # Log failure to MLflow if enabled
        if mlflow_manager:
            mlflow_manager.log_param("error_message", str(e))
            mlflow_manager.log_metric("total_processing_time_seconds", (datetime.now() - pipeline_start_time).total_seconds())
            mlflow_manager.end_run(status="FAILED")
        
        print(f"Error: {e}")
        print("See logs/data_processing.log for details")
        exit(1)


if __name__ == "__main__":
    main()