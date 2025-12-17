import logging
from dotenv import load_dotenv
from utils.app_insights import configure_app_insights
from registration_processor import RegistrationProcessor
from scan_processor import ScanProcessor
from session_processor import SessionProcessor
from neo4j_visitor_processor import Neo4jVisitorProcessor
from neo4j_session_processor import Neo4jSessionProcessor
from neo4j_job_stream_processor import Neo4jJobStreamProcessor
from neo4j_specialization_stream_processor import Neo4jSpecializationStreamProcessor
from neo4j_visitor_relationship_processor import Neo4jVisitorRelationshipProcessor
from session_embedding_processor import SessionEmbeddingProcessor
from session_recommendation_processor import SessionRecommendationProcessor

load_dotenv()
configure_app_insights(service_name="pa_local_pipeline")


# Replace the run_registration_processing function in pipeline.py:

def run_registration_processing(config):
    """
    Run the registration data processing step.
    Enhanced version with better veterinary-specific function handling.

    Args:
        config: Configuration dictionary

    Returns:
        RegistrationProcessor instance
    """
    logger = logging.getLogger(__name__)
    
    # Check if this is a veterinary event and apply vet-specific functions
    event_config = config.get("event", {})
    main_event_name = event_config.get("main_event_name", "").lower()
    
    logger.info(f"Initializing registration processor for event: {main_event_name}")
    reg_processor = RegistrationProcessor(config)
    
    # Apply event-specific enhancements based on event type
    if main_event_name in ["bva", "lva"]:
        logger.info("Detected veterinary event - applying veterinary-specific processing functions")
        try:
            from utils import vet_specific_functions
            
            # Apply veterinary-specific methods
            vet_specific_functions.add_vet_specific_methods(reg_processor)
            
            # Verify that the functions were applied correctly
            if vet_specific_functions.verify_vet_functions_applied(reg_processor):
                logger.info("✅ Veterinary-specific processing functions successfully applied and verified")
                reg_processor.logger.info("Veterinary-specific functions are active for this processing session")
            else:
                logger.error("❌ Veterinary-specific function verification failed")
                raise RuntimeError("Failed to properly apply veterinary-specific functions")
            
        except ImportError as e:
            logger.warning(f"Could not load vet-specific functions: {e}")
            logger.warning("Proceeding with generic processing logic")
        except Exception as e:
            logger.error(f"Error applying vet-specific functions: {e}", exc_info=True)
            logger.warning("Proceeding with generic processing logic")
    else:
        logger.info(f"Using generic processing functions for event type: {main_event_name}")
        reg_processor.logger.info("Generic event processing functions are active for this processing session")
        
        # Ensure vet-specific flag is not set for non-vet events
        if hasattr(reg_processor, '_vet_specific_active'):
            delattr(reg_processor, '_vet_specific_active')
    
    # Log the final state
    is_vet_specific = hasattr(reg_processor, '_vet_specific_active') and reg_processor._vet_specific_active
    logger.info(f"Registration processor initialized with {'veterinary-specific' if is_vet_specific else 'generic'} functions")
    
    # Process the data
    logger.info("Starting registration data processing")
    reg_processor.process()
    logger.info("Registration data processing completed")
    
    return reg_processor


def run_scan_processing(config):
    """
    Run the scan data processing step.

    Args:
        config: Configuration dictionary

    Returns:
        ScanProcessor instance
    """
    scan_processor = ScanProcessor(config)
    scan_processor.process()
    return scan_processor


def run_session_processing(config):
    """
    Run the session data processing step.

    Args:
        config: Configuration dictionary

    Returns:
        SessionProcessor instance
    """
    session_processor = SessionProcessor(config)
    session_processor.process()
    return session_processor


def run_neo4j_processing(config, create_only_new=True, steps_to_run=None):
    """
    Run all Neo4j data processing steps.

    Args:
        config: Configuration dictionary
        create_only_new: If True, only create new nodes if they don't already exist
        steps_to_run: List of step numbers to run (if None, run all enabled steps)

    Returns:
        Dictionary of processor instances
    """
    logger = logging.getLogger(__name__)
    processors = {}

    # Check if specific steps are requested
    run_all = steps_to_run is None

    # Helper to decide logging for non-selected steps
    def _log_non_selected(step_num: int, name: str):
        # Only log an INFO skip if explicitly disabled in config; otherwise keep noise down
        enabled_map = {
            4: config.get("pipeline_steps", {}).get("neo4j_visitor_processing", True),
            5: config.get("pipeline_steps", {}).get("neo4j_session_processing", True),
            6: config.get("pipeline_steps", {}).get("neo4j_job_stream_processing", True),
            7: config.get("pipeline_steps", {}).get("neo4j_specialization_stream_processing", True),
            8: config.get("pipeline_steps", {}).get("neo4j_visitor_relationship_processing", True),
            9: config.get("pipeline_steps", {}).get("session_embedding_processing", True),
            10: config.get("pipeline_steps", {}).get("session_recommendation_processing", True),
        }
        if not enabled_map.get(step_num, True):
            logger.info(f"Skipping step {step_num}: {name} (disabled in config)")
        elif run_all:
            # run_all handled elsewhere, so this path shouldn't occur
            logger.debug(f"Step {step_num} not executed (unexpected path)")
        else:
            # Not selected in steps_to_run: keep log level low
            logger.debug(f"Step {step_num} not in selected steps list; not executing {name}")

    # Neo4j visitor processor (Step 4)
    if (run_all or 4 in steps_to_run):
        if config.get("pipeline_steps", {}).get("neo4j_visitor_processing", True):
            logger.info("Starting step 4: Neo4j visitor data processing")
            neo4j_visitor_processor = Neo4jVisitorProcessor(config)
            neo4j_visitor_processor.process(create_only_new=create_only_new)
            processors["neo4j_visitor_processor"] = neo4j_visitor_processor
            logger.info("Completed step 4: Neo4j visitor data processing")
        else:
            logger.info("Skipping step 4: Neo4j visitor data processing (disabled in config)")
    else:
        _log_non_selected(4, "Neo4j visitor data processing")

    # Neo4j session processor (Step 5)
    if (run_all or 5 in steps_to_run):
        if config.get("pipeline_steps", {}).get("neo4j_session_processing", True):
            logger.info("Starting step 5: Neo4j session data processing")
            neo4j_session_processor = Neo4jSessionProcessor(config)
            neo4j_session_processor.process(create_only_new=create_only_new)
            processors["neo4j_session_processor"] = neo4j_session_processor
            logger.info("Completed step 5: Neo4j session data processing")
        else:
            logger.info("Skipping step 5: Neo4j session data processing (disabled in config)")
    else:
        _log_non_selected(5, "Neo4j session data processing")

    # Neo4j job stream processor (Step 6)
    if (run_all or 6 in steps_to_run):
        if config.get("pipeline_steps", {}).get("neo4j_job_stream_processing", True):
            logger.info("Starting step 6: Neo4j job to stream relationship processing")
            neo4j_job_stream_processor = Neo4jJobStreamProcessor(config)
            neo4j_job_stream_processor.process(create_only_new=create_only_new)
            processors["neo4j_job_stream_processor"] = neo4j_job_stream_processor
            logger.info("Completed step 6: Neo4j job to stream relationship processing")
        else:
            logger.info("Skipping step 6: Neo4j job to stream relationship processing (disabled in config)")
    else:
        _log_non_selected(6, "Neo4j job to stream relationship processing")

    # Neo4j specialization stream processor (Step 7)
    if (run_all or 7 in steps_to_run):
        if config.get("pipeline_steps", {}).get("neo4j_specialization_stream_processing", True):
            logger.info("Starting step 7: Neo4j specialization to stream relationship processing")
            neo4j_specialization_stream_processor = Neo4jSpecializationStreamProcessor(config)
            neo4j_specialization_stream_processor.process(create_only_new=create_only_new)
            processors["neo4j_specialization_stream_processor"] = neo4j_specialization_stream_processor
            logger.info("Completed step 7: Neo4j specialization to stream relationship processing")
        else:
            logger.info("Skipping step 7: Neo4j specialization to stream relationship processing (disabled in config)")
    else:
        _log_non_selected(7, "Neo4j specialization to stream relationship processing")

    # Neo4j visitor relationship processor (Step 8)
    if (run_all or 8 in steps_to_run):
        if config.get("pipeline_steps", {}).get("neo4j_visitor_relationship_processing", True):
            logger.info("Starting step 8: Neo4j visitor relationship processing")
            neo4j_visitor_relationship_processor = Neo4jVisitorRelationshipProcessor(config)
            neo4j_visitor_relationship_processor.process(create_only_new=create_only_new)
            processors["neo4j_visitor_relationship_processor"] = neo4j_visitor_relationship_processor
            logger.info("Completed step 8: Neo4j visitor relationship processing")
        else:
            logger.info("Skipping step 8: Neo4j visitor relationship processing (disabled in config)")
    else:
        _log_non_selected(8, "Neo4j visitor relationship processing")

    # Session embedding processor (Step 9)
    if (run_all or 9 in steps_to_run):
        if config.get("pipeline_steps", {}).get("session_embedding_processing", True):
            logger.info("Starting step 9: Session embedding processing")
            session_embedding_processor = SessionEmbeddingProcessor(config)
            session_embedding_processor.process(create_only_new=create_only_new)
            processors["session_embedding_processor"] = session_embedding_processor
            logger.info("Completed step 9: Session embedding processing")
        else:
            logger.info("Skipping step 9: Session embedding processing (disabled in config)")
    else:
        _log_non_selected(9, "Session embedding processing")

    # Session recommendation processor (Step 10)
    if (run_all or 10 in steps_to_run):
        if config.get("pipeline_steps", {}).get("session_recommendation_processing", True):
            logger.info("Starting step 10: Session recommendation processing")
            session_recommendation_processor = SessionRecommendationProcessor(config)
            session_recommendation_processor.process(create_only_new=create_only_new)
            processors["session_recommendation_processor"] = session_recommendation_processor
            logger.info("Completed step 10: Session recommendation processing")
        else:
            logger.info("Skipping step 10: Session recommendation processing (disabled in config)")
    else:
        _log_non_selected(10, "Session recommendation processing")

    return processors


def run_session_recommendation_processing(config, create_only_new=True):
    """
    Run session recommendation processing step.

    Args:
        config: Configuration dictionary
        create_only_new: If True, only create new recommendations if they don't already exist

    Returns:
        SessionRecommendationProcessor instance
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting session recommendation processing")

    session_recommendation_processor = SessionRecommendationProcessor(config)
    session_recommendation_processor.process(create_only_new=create_only_new)

    logger.info("Completed session recommendation processing")
    return session_recommendation_processor