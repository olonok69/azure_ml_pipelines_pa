"""
Enhanced Summary utilities for generating and printing pipeline statistics.

This module provides functions to generate comprehensive summaries of the data processing pipeline,
including registration, scan, session, and Neo4j processing statistics.

ENHANCED: Compatible with both old and new processor statistics formats.
"""

import json
import logging
from datetime import datetime


def generate_and_save_summary(processors, skip_neo4j=False):
    """
    Generate a comprehensive summary of the data processing pipeline results.
    
    ENHANCED: Compatible with both old and new processor statistics formats.

    Args:
        processors: Dictionary of processor instances
        skip_neo4j: Boolean indicating if Neo4j processing was skipped

    Returns:
        dict: Summary statistics
    """
    logger = logging.getLogger(__name__)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": "enhanced",
        "processing_mode": "generic" if not skip_neo4j else "data_only",
        "processors_run": list(processors.keys()),
    }

    # Get key processors
    reg_processor = processors.get("registration_processor")
    scan_processor = processors.get("scan_processor")
    session_processor = processors.get("session_processor")

    # Create summary statistics for registration data if available
    if reg_processor and hasattr(reg_processor, "df_reg_demo_this"):
        reg_summary = {
            "total_registrations": {
                "this_year": len(reg_processor.df_reg_demo_this),
                "last_year_bva": len(reg_processor.df_reg_demo_last_bva)
                if hasattr(reg_processor, "df_reg_demo_last_bva")
                else 0,
                "last_year_lva": len(reg_processor.df_reg_demo_last_lva)
                if hasattr(reg_processor, "df_reg_demo_last_lva")
                else 0,
            },
            "unique_countries": len(
                reg_processor.df_reg_demo_this["Country"].unique()
            )
            if "Country" in reg_processor.df_reg_demo_this.columns
            else 0,
            "unique_organizations": len(
                reg_processor.df_reg_demo_this["Company"].unique()
            )
            if "Company" in reg_processor.df_reg_demo_this.columns
            else 0,
        }

        # Add job role distribution if available
        job_role_columns = [
            "job_role",
            "JobTitle",
            "what_is_your_job_role",
        ]
        job_role_found = False
        for col in job_role_columns:
            if col in reg_processor.df_reg_demo_this.columns:
                reg_summary["job_roles"] = reg_processor.df_reg_demo_this[col].value_counts().to_dict()
                logger.info(f"Using job role column: {col}")
                job_role_found = True
                break
        
        if not job_role_found:
            logger.info("No job role column found - skipping job role summary")
            reg_summary["job_roles"] = {}

        # Add specialization distribution if available
        specialization_columns = [
            "what_type_does_your_practice_specialise_in",
            "what_areas_do_you_specialise_in",
            "what_best_describes_your_specialism",
        ]
        specialization_found = False
        for col in specialization_columns:
            if col in reg_processor.df_reg_demo_this.columns:
                reg_summary["specializations"] = reg_processor.df_reg_demo_this[col].value_counts().to_dict()
                logger.info(f"Using specialization column: {col}")
                specialization_found = True
                break
        
        if not specialization_found:
            logger.info("No specialization column found - skipping specialization summary")
            reg_summary["specializations"] = {}
        
        summary["registration"] = reg_summary

    # Create summary statistics for scan data if available
    if (
        scan_processor
        and hasattr(scan_processor, "enhanced_seminars_df_bva")
        and hasattr(scan_processor, "enhanced_seminars_df_lva")
    ):
        scan_summary = {
            "total_scans": {
                "last_year_bva": len(scan_processor.enhanced_seminars_df_bva),
                "last_year_lva": len(scan_processor.enhanced_seminars_df_lva),
            },
            "unique_seminars": {
                "last_year_bva": len(
                    scan_processor.seminars_scans_past_enhanced_bva[
                        "Seminar Name"
                    ].unique()
                )
                if hasattr(scan_processor, "seminars_scans_past_enhanced_bva")
                else 0,
                "last_year_lva": len(
                    scan_processor.seminars_scans_past_enhanced_lva[
                        "Seminar Name"
                    ].unique()
                )
                if hasattr(scan_processor, "seminars_scans_past_enhanced_lva")
                else 0,
            },
            "unique_attendees": {
                "last_year_bva": len(
                    scan_processor.enhanced_seminars_df_bva["Badge Id"].unique()
                )
                if "Badge Id" in scan_processor.enhanced_seminars_df_bva.columns
                else 0,
                "last_year_lva": len(
                    scan_processor.enhanced_seminars_df_lva["Badge Id"].unique()
                )
                if "Badge Id" in scan_processor.enhanced_seminars_df_lva.columns
                else 0,
            },
        }
        summary["scan"] = scan_summary

    # Create summary statistics for session data if available
    if (
        session_processor
        and hasattr(session_processor, "session_this_filtered_valid_cols")
        and hasattr(session_processor, "streams_catalog")
    ):
        session_summary = {
            "total_sessions": {
                "this_year": len(session_processor.session_this_filtered_valid_cols),
                "last_year_bva": len(
                    session_processor.session_last_filtered_valid_cols_bva
                )
                if hasattr(session_processor, "session_last_filtered_valid_cols_bva")
                else 0,
                "last_year_lva": len(
                    session_processor.session_last_filtered_valid_cols_lva
                )
                if hasattr(session_processor, "session_last_filtered_valid_cols_lva")
                else 0,
            },
            # FIXED: Use unique_streams instead of streams
            "unique_streams": len(session_processor.unique_streams) if hasattr(session_processor, "unique_streams") else 0,
            "stream_categories": list(session_processor.streams_catalog.keys()) if session_processor.streams_catalog else [],
        }

        if hasattr(session_processor, "backfill_metrics"):
            session_summary["missing_stream_backfill"] = session_processor.backfill_metrics
        summary["session"] = session_summary

    # Add Neo4j statistics if available and Neo4j processing wasn't skipped
    if not skip_neo4j:
        add_neo4j_statistics(summary, processors)

    # Add session recommendation statistics if available
    session_recommendation_processor = processors.get(
        "session_recommendation_processor"
    )
    if session_recommendation_processor and hasattr(
        session_recommendation_processor, "statistics"
    ):
        session_recommendation_summary = session_recommendation_processor.statistics
        summary["session_recommendation"] = session_recommendation_summary

    # Save summary as JSON
    with open("logs/processing_summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=str)

    # Log and print summary information
    print_summary_statistics(summary, skip_neo4j, reg_processor)

    return summary


def add_neo4j_statistics(summary, processors):
    """
    Add Neo4j-related statistics to the summary.
    
    ENHANCED: Compatible with both old and new processor statistics formats.

    Args:
        summary: Summary dictionary to update
        processors: Dictionary of processor instances
    """
    # Add Neo4j visitor statistics if available
    neo4j_visitor_processor = processors.get("neo4j_visitor_processor")
    if neo4j_visitor_processor and hasattr(neo4j_visitor_processor, "statistics"):
        neo4j_visitor_summary = {
            "nodes_created": neo4j_visitor_processor.statistics["nodes_created"],
            "nodes_skipped": neo4j_visitor_processor.statistics["nodes_skipped"],
            "total_nodes_created": sum(
                neo4j_visitor_processor.statistics["nodes_created"].values()
            ),
            "total_nodes_skipped": sum(
                neo4j_visitor_processor.statistics["nodes_skipped"].values()
            ),
        }
        summary["neo4j_visitor"] = neo4j_visitor_summary

    # Add Neo4j session statistics if available
    neo4j_session_processor = processors.get("neo4j_session_processor")
    if neo4j_session_processor and hasattr(neo4j_session_processor, "statistics"):
        neo4j_session_summary = {
            "nodes_created": neo4j_session_processor.statistics["nodes_created"],
            "nodes_skipped": neo4j_session_processor.statistics["nodes_skipped"],
            "relationships_created": neo4j_session_processor.statistics[
                "relationships_created"
            ],
            "relationships_skipped": neo4j_session_processor.statistics[
                "relationships_skipped"
            ],
            "total_nodes_created": sum(
                neo4j_session_processor.statistics["nodes_created"].values()
            ),
            "total_nodes_skipped": sum(
                neo4j_session_processor.statistics["nodes_skipped"].values()
            ),
            "total_relationships_created": sum(
                neo4j_session_processor.statistics["relationships_created"].values()
            ),
            "total_relationships_skipped": sum(
                neo4j_session_processor.statistics["relationships_skipped"].values()
            ),
        }
        summary["neo4j_session"] = neo4j_session_summary

    # Add Neo4j job stream statistics if available (ENHANCED)
    neo4j_job_stream_processor = processors.get("neo4j_job_stream_processor")
    if neo4j_job_stream_processor and hasattr(neo4j_job_stream_processor, "statistics"):
        neo4j_job_stream_summary = neo4j_job_stream_processor.statistics.copy()
        summary["neo4j_job_stream"] = neo4j_job_stream_summary

    # Add Neo4j specialization stream statistics if available (ENHANCED)
    neo4j_specialization_stream_processor = processors.get(
        "neo4j_specialization_stream_processor"
    )
    if neo4j_specialization_stream_processor and hasattr(
        neo4j_specialization_stream_processor, "statistics"
    ):
        neo4j_specialization_stream_summary = (
            neo4j_specialization_stream_processor.statistics.copy()
        )
        
        # ENHANCED: Handle both old and new statistics formats
        relationships_created = neo4j_specialization_stream_summary.get("relationships_created", 0)
        relationships_skipped = neo4j_specialization_stream_summary.get("relationships_skipped", 0)
        visitor_nodes_processed = neo4j_specialization_stream_summary.get("visitor_nodes_processed", {})
        
        # If relationships_created is a dict (old format), calculate totals
        if isinstance(relationships_created, dict):
            neo4j_specialization_stream_summary["total_relationships_created"] = sum(
                relationships_created.values()
            )
        else:
            # If it's already an int (new format), use as is
            neo4j_specialization_stream_summary["total_relationships_created"] = relationships_created
        
        # If relationships_skipped is a dict (old format), calculate totals
        if isinstance(relationships_skipped, dict):
            neo4j_specialization_stream_summary["total_relationships_skipped"] = sum(
                relationships_skipped.values()
            )
        else:
            # If it's already an int (new format), use as is
            neo4j_specialization_stream_summary["total_relationships_skipped"] = relationships_skipped
        
        # Handle visitor_nodes_processed
        if isinstance(visitor_nodes_processed, dict):
            neo4j_specialization_stream_summary["total_visitor_nodes_processed"] = sum(
                visitor_nodes_processed.values()
            )
        else:
            neo4j_specialization_stream_summary["total_visitor_nodes_processed"] = visitor_nodes_processed
        
        summary["neo4j_specialization_stream"] = neo4j_specialization_stream_summary

    # Add Neo4j visitor relationship statistics if available (ENHANCED)
    neo4j_visitor_relationship_processor = processors.get(
        "neo4j_visitor_relationship_processor"
    )
    if neo4j_visitor_relationship_processor and hasattr(
        neo4j_visitor_relationship_processor, "statistics"
    ):
        neo4j_visitor_relationship_summary = (
            neo4j_visitor_relationship_processor.statistics.copy()
        )
        
        # Handle both old and new formats
        relationships_created = neo4j_visitor_relationship_summary.get("relationships_created", {})
        relationships_skipped = neo4j_visitor_relationship_summary.get("relationships_skipped", {})
        
        if isinstance(relationships_created, dict):
            neo4j_visitor_relationship_summary["total_relationships_created"] = sum(
                relationships_created.values()
            )
        else:
            neo4j_visitor_relationship_summary["total_relationships_created"] = relationships_created
            
        if isinstance(relationships_skipped, dict):
            neo4j_visitor_relationship_summary["total_relationships_skipped"] = sum(
                relationships_skipped.values()
            )
        else:
            neo4j_visitor_relationship_summary["total_relationships_skipped"] = relationships_skipped
            
        summary["neo4j_visitor_relationship"] = neo4j_visitor_relationship_summary

    # Add Session Embedding statistics if available
    session_embedding_processor = processors.get("session_embedding_processor")
    if session_embedding_processor and hasattr(
        session_embedding_processor, "statistics"
    ):
        session_embedding_summary = session_embedding_processor.statistics
        summary["session_embedding"] = session_embedding_summary


def print_summary_statistics(summary, skip_neo4j, reg_processor=None):
    """
    Print summary statistics to console and log key information.
    
    ENHANCED: Compatible with both old and new processor statistics formats.

    Args:
        summary: Summary dictionary
        skip_neo4j: Boolean indicating if Neo4j processing was skipped
        reg_processor: Registration processor instance (optional)
    """
    logger = logging.getLogger(__name__)

    # Get event configuration for printing
    main_event_name = "BVA"
    secondary_event_name = "LVA"
    
    # Try to get event names from registration processor
    if reg_processor and hasattr(reg_processor, 'config'):
        event_config = reg_processor.config.get('event', {})
        main_event_name = event_config.get('main_event_name', 'BVA').upper()
        secondary_event_name = event_config.get('secondary_event_name', 'LVA').upper()

    # Print header
    print("\n" + "=" * 60)
    print("PIPELINE PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Processing completed at: {summary['timestamp']}")
    print(f"Pipeline version: {summary['pipeline_version']}")
    print(f"Processing mode: {summary['processing_mode']}")
    print(f"Processors run: {', '.join(summary['processors_run'])}")

    # Print registration summary
    if "registration" in summary:
        reg_summary = summary["registration"]
        logger.info(
            f"Total registrations this year: {reg_summary['total_registrations']['this_year']}"
        )
        logger.info(
            f"Total registrations last year {main_event_name}: {reg_summary['total_registrations']['last_year_bva']}"
        )
        logger.info(
            f"Total registrations last year {secondary_event_name}: {reg_summary['total_registrations']['last_year_lva']}"
        )
        logger.info(f"Unique countries: {reg_summary['unique_countries']}")
        logger.info(f"Unique organizations: {reg_summary['unique_organizations']}")

        print("\nRegistration Summary:")
        print(
            f"Total registrations this year: {reg_summary['total_registrations']['this_year']}"
        )
        print(
            f"Total registrations last year {main_event_name}: {reg_summary['total_registrations']['last_year_bva']}"
        )
        print(
            f"Total registrations last year {secondary_event_name}: {reg_summary['total_registrations']['last_year_lva']}"
        )
        print(f"Unique countries represented: {reg_summary['unique_countries']}")
        print(f"Unique organizations: {reg_summary['unique_organizations']}")

        if reg_summary.get("job_roles"):
            print("\nTop 5 Job Roles:")
            for i, (role, count) in enumerate(list(reg_summary["job_roles"].items())[:5]):
                print(f"  {i+1}. {role}: {count}")

        if reg_summary.get("specializations"):
            print("\nTop 5 Specializations:")
            for i, (spec, count) in enumerate(list(reg_summary["specializations"].items())[:5]):
                print(f"  {i+1}. {spec}: {count}")

    # Print scan summary
    if "scan" in summary:
        scan_summary = summary["scan"]
        logger.info(
            f"Total scans last year {main_event_name}: {scan_summary['total_scans']['last_year_bva']}"
        )
        logger.info(
            f"Total scans last year {secondary_event_name}: {scan_summary['total_scans']['last_year_lva']}"
        )
        logger.info(
            f"Unique seminars last year {main_event_name}: {scan_summary['unique_seminars']['last_year_bva']}"
        )
        logger.info(
            f"Unique seminars last year {secondary_event_name}: {scan_summary['unique_seminars']['last_year_lva']}"
        )

        print("\nScan Summary:")
        print(
            f"Total scans last year {main_event_name}: {scan_summary['total_scans']['last_year_bva']}"
        )
        print(
            f"Total scans last year {secondary_event_name}: {scan_summary['total_scans']['last_year_lva']}"
        )
        print(
            f"Unique seminars last year {main_event_name}: {scan_summary['unique_seminars']['last_year_bva']}"
        )
        print(
            f"Unique seminars last year {secondary_event_name}: {scan_summary['unique_seminars']['last_year_lva']}"
        )
        print(
            f"Unique seminar attendees last year {main_event_name}: {scan_summary['unique_attendees']['last_year_bva']}"
        )
        print(
            f"Unique seminar attendees last year {secondary_event_name}: {scan_summary['unique_attendees']['last_year_lva']}"
        )

    # Print session summary
    if "session" in summary:
        session_summary = summary["session"]
        logger.info(
            f"Total sessions this year: {session_summary['total_sessions']['this_year']}"
        )
        logger.info(
            f"Total sessions last year {main_event_name}: {session_summary['total_sessions']['last_year_bva']}"
        )
        logger.info(
            f"Total sessions last year {secondary_event_name}: {session_summary['total_sessions']['last_year_lva']}"
        )
        logger.info(
            f"Total unique stream categories: {session_summary['unique_streams']}"
        )

        print("\nSession Summary:")
        print(
            f"Total sessions this year: {session_summary['total_sessions']['this_year']}"
        )
        print(
            f"Total sessions last year {main_event_name}: {session_summary['total_sessions']['last_year_bva']}"
        )
        print(
            f"Total sessions last year {secondary_event_name}: {session_summary['total_sessions']['last_year_lva']}"
        )
        print(f"Total unique stream categories: {session_summary['unique_streams']}")
        print("\nTop 5 Stream Categories:")
        for i, stream in enumerate(session_summary["stream_categories"][:5]):
            print(f"  {i+1}. {stream}")

    # Print Neo4j statistics if available
    if not skip_neo4j:
        print_neo4j_statistics(summary, main_event_name, secondary_event_name)

    # Print session recommendation summary
    if "session_recommendation" in summary:
        rec_summary = summary["session_recommendation"]
        total_visitors_processed = rec_summary.get("total_visitors_processed", 0)
        visitors_with_recommendations = rec_summary.get(
            "visitors_with_recommendations", 0
        )
        total_recommendations_generated = rec_summary.get(
            "total_recommendations_generated", 0
        )
        unique_recommended_sessions = rec_summary.get(
            "unique_recommended_sessions", 0
        )
        errors_count = rec_summary.get("errors", 0)

        logger.info(
            f"Total visitors processed for recommendations: {total_visitors_processed}"
        )
        logger.info(
            f"Visitors with successful recommendations: {visitors_with_recommendations}"
        )
        logger.info(
            f"Total recommendations generated: {total_recommendations_generated}"
        )
        logger.info(
            f"Unique sessions recommended: {unique_recommended_sessions}"
        )
        if errors_count > 0:
            logger.info(f"Errors encountered: {errors_count}")

        print("\nSession Recommendation Summary:")
        print(f"Total visitors processed: {total_visitors_processed}")
        print(
            f"Visitors with successful recommendations: {visitors_with_recommendations}"
        )
        print(
            f"Total recommendations generated: {total_recommendations_generated}"
        )
        print(f"Unique sessions recommended: {unique_recommended_sessions}")
        if errors_count > 0:
            print(f"Errors encountered: {errors_count}")

    # Print summary location
    print(f"\n📄 Detailed summary saved to: logs/processing_summary.json")
    print("=" * 60)


def print_neo4j_statistics(summary, main_event_name, secondary_event_name):
    """
    Print Neo4j-related statistics.
    
    ENHANCED: Compatible with both old and new processor statistics formats.

    Args:
        summary: Summary dictionary
        main_event_name: Name of main event (e.g., BVA)
        secondary_event_name: Name of secondary event (e.g., LVA)
    """
    logger = logging.getLogger(__name__)

    # Print Neo4j visitor statistics
    if "neo4j_visitor" in summary:
        neo4j_visitor_summary = summary["neo4j_visitor"]
        print("\nNeo4j Visitor Node Summary:")
        print(
            f"Total visitor nodes created: {neo4j_visitor_summary['total_nodes_created']}"
        )
        print(
            f"Total visitor nodes skipped: {neo4j_visitor_summary['total_nodes_skipped']}"
        )
        print("\nDetailed Visitor Node Statistics:")
        print(
            f"  This year visitors: {neo4j_visitor_summary['nodes_created']['visitor_this_year']} created, {neo4j_visitor_summary['nodes_skipped']['visitor_this_year']} skipped"
        )
        print(
            f"  Last year {main_event_name} visitors: {neo4j_visitor_summary['nodes_created']['visitor_last_year_bva']} created, {neo4j_visitor_summary['nodes_skipped']['visitor_last_year_bva']} skipped"
        )
        print(
            f"  Last year {secondary_event_name} visitors: {neo4j_visitor_summary['nodes_created']['visitor_last_year_lva']} created, {neo4j_visitor_summary['nodes_skipped']['visitor_last_year_lva']} skipped"
        )

    # Print Neo4j session statistics
    if "neo4j_session" in summary:
        neo4j_session_summary = summary["neo4j_session"]
        print("\nNeo4j Session Node Summary:")
        print(
            f"Total session nodes created: {neo4j_session_summary['total_nodes_created']}"
        )
        print(
            f"Total session nodes skipped: {neo4j_session_summary['total_nodes_skipped']}"
        )
        print(
            f"Total session-stream relationships created: {neo4j_session_summary['total_relationships_created']}"
        )
        print(
            f"Total session-stream relationships skipped: {neo4j_session_summary['total_relationships_skipped']}"
        )
        print("\nDetailed Session Node Statistics:")
        print(
            f"  This year sessions: {neo4j_session_summary['nodes_created']['sessions_this_year']} created, {neo4j_session_summary['nodes_skipped']['sessions_this_year']} skipped"
        )
        print(
            f"  Past year sessions: {neo4j_session_summary['nodes_created']['sessions_past_year']} created, {neo4j_session_summary['nodes_skipped']['sessions_past_year']} skipped"
        )
        print(
            f"  Session-Stream relationships this year: {neo4j_session_summary['relationships_created']['sessions_this_year_has_stream']} created, {neo4j_session_summary['relationships_skipped']['sessions_this_year_has_stream']} skipped"
        )
        print(
            f"  Session-Stream relationships past year: {neo4j_session_summary['relationships_created']['sessions_past_year_has_stream']} created, {neo4j_session_summary['relationships_skipped']['sessions_past_year_has_stream']} skipped"
        )

    # Print Neo4j job stream statistics (ENHANCED)
    if "neo4j_job_stream" in summary:
        neo4j_job_stream_summary = summary["neo4j_job_stream"]
        print("\nNeo4j Job-Stream Relationship Summary:")
        
        # Handle skipped processing
        if neo4j_job_stream_summary.get("processing_skipped", False):
            skip_reason = neo4j_job_stream_summary.get("skip_reason", "Unknown reason")
            print(f"Processing skipped: {skip_reason}")
        else:
            print(
                f"Total job-stream relationships created: {neo4j_job_stream_summary['relationships_created']}"
            )
            print(
                f"Total job-stream relationships skipped: {neo4j_job_stream_summary['relationships_skipped']}"
            )
            print("\nDetailed Job-Stream Statistics:")
            print(
                f"  Visitor nodes processed: {neo4j_job_stream_summary['visitor_nodes_processed']}"
            )
            print(f"  Job roles matched: {neo4j_job_stream_summary['job_roles_processed']}")
            print(
                f"  Stream matches found: {neo4j_job_stream_summary['stream_matches_found']}"
            )

    # Print Neo4j specialization stream statistics (ENHANCED)
    if "neo4j_specialization_stream" in summary:
        neo4j_specialization_stream_summary = summary["neo4j_specialization_stream"]
        print("\nNeo4j Specialization-Stream Relationship Summary:")
        
        # Handle skipped processing
        if neo4j_specialization_stream_summary.get("processing_skipped", False):
            skip_reason = neo4j_specialization_stream_summary.get("skip_reason", "Unknown reason")
            print(f"Processing skipped: {skip_reason}")
        else:
            # Handle both old and new formats
            total_created = neo4j_specialization_stream_summary.get("total_relationships_created", 
                neo4j_specialization_stream_summary.get("relationships_created", 0))
            total_skipped = neo4j_specialization_stream_summary.get("total_relationships_skipped",
                neo4j_specialization_stream_summary.get("relationships_skipped", 0))
            
            print(f"Total specialization-stream relationships created: {total_created}")
            print(f"Total specialization-stream relationships skipped: {total_skipped}")
            
            print("\nDetailed Specialization-Stream Statistics:")
            visitor_nodes_processed = neo4j_specialization_stream_summary.get("visitor_nodes_processed", {})
            
            if isinstance(visitor_nodes_processed, dict):
                print(f"  Visitor nodes processed:")
                # Handle both old and new key formats
                this_year_key = next((key for key in visitor_nodes_processed.keys() 
                                    if 'this_year' in key.lower()), None)
                bva_key = next((key for key in visitor_nodes_processed.keys() 
                              if 'bva' in key.lower() and 'last' in key.lower()), None)
                lva_key = next((key for key in visitor_nodes_processed.keys() 
                              if 'lva' in key.lower() and 'last' in key.lower()), None)
                
                if this_year_key:
                    print(f"    This year: {visitor_nodes_processed[this_year_key]}")
                if bva_key:
                    print(f"    Last year {main_event_name}: {visitor_nodes_processed[bva_key]}")
                if lva_key:
                    print(f"    Last year {secondary_event_name}: {visitor_nodes_processed[lva_key]}")
            else:
                print(f"  Total visitor nodes processed: {visitor_nodes_processed}")
            
            specializations_processed = neo4j_specialization_stream_summary.get("specializations_processed", 0)
            stream_matches_found = neo4j_specialization_stream_summary.get("stream_matches_found", 0)
            
            print(f"  Specializations processed: {specializations_processed}")
            print(f"  Stream matches found: {stream_matches_found}")

    # Print Neo4j visitor relationship statistics (ENHANCED)
    if "neo4j_visitor_relationship" in summary:
        neo4j_visitor_relationship_summary = summary["neo4j_visitor_relationship"]
        print("\nNeo4j Visitor Relationship Summary:")
        
        total_created = neo4j_visitor_relationship_summary.get("total_relationships_created", 0)
        total_skipped = neo4j_visitor_relationship_summary.get("total_relationships_skipped", 0)
        
        print(f"Total visitor relationships created: {total_created}")
        print(f"Total visitor relationships skipped: {total_skipped}")
        
        # Print detailed stats if available
        relationships_created = neo4j_visitor_relationship_summary.get("relationships_created", {})
        relationships_skipped = neo4j_visitor_relationship_summary.get("relationships_skipped", {})
        
        if isinstance(relationships_created, dict):
            print("\nDetailed Visitor Relationship Statistics:")
            same_visitor_bva = relationships_created.get("same_visitor_bva", 0)
            same_visitor_lva = relationships_created.get("same_visitor_lva", 0)
            attended_session_bva = relationships_created.get("attended_session_bva", 0)
            attended_session_lva = relationships_created.get("attended_session_lva", 0)
            
            same_visitor_bva_skipped = relationships_skipped.get("same_visitor_bva", 0)
            same_visitor_lva_skipped = relationships_skipped.get("same_visitor_lva", 0)
            attended_session_bva_skipped = relationships_skipped.get("attended_session_bva", 0)
            attended_session_lva_skipped = relationships_skipped.get("attended_session_lva", 0)
            
            print(f"  Same_Visitor relationships for {main_event_name}: {same_visitor_bva} created, {same_visitor_bva_skipped} skipped")
            print(f"  Same_Visitor relationships for {secondary_event_name}: {same_visitor_lva} created, {same_visitor_lva_skipped} skipped")
            print(f"  attended_session relationships for {main_event_name}: {attended_session_bva} created, {attended_session_bva_skipped} skipped")
            print(f"  attended_session relationships for {secondary_event_name}: {attended_session_lva} created, {attended_session_lva_skipped} skipped")

    # Print session embedding statistics
    if "session_embedding" in summary:
        session_embedding_summary = summary["session_embedding"]
        logger.info(
            f"Total sessions processed for embeddings: {session_embedding_summary['total_sessions_processed']}"
        )
        logger.info(
            f"Sessions with embeddings created: {session_embedding_summary['sessions_with_embeddings']}"
        )

        print("\nSession Embedding Summary:")
        print(
            f"Total sessions processed: {session_embedding_summary['total_sessions_processed']}"
        )
        print(
            f"Sessions with embeddings created: {session_embedding_summary['sessions_with_embeddings']}"
        )
        print(
            f"Sessions with stream descriptions included: {session_embedding_summary['sessions_with_stream_descriptions']}"
        )
        print("\nDetailed Session Embedding Statistics:")
        print(
            f"  Sessions this year: {session_embedding_summary['sessions_by_type']['sessions_this_year']}"
        )
        print(
            f"  Sessions past year: {session_embedding_summary['sessions_by_type']['sessions_past_year']}"
        )
        if session_embedding_summary.get("errors", 0) > 0:
            print(f"  Errors encountered: {session_embedding_summary['errors']}")


def print_neo4j_usage_info():
    """
    Print Neo4j usage information and example queries.
    """
    print("\n" + "=" * 60)
    print("NEO4J DATABASE USAGE INFORMATION")
    print("=" * 60)
    print("The following nodes and relationships have been created in Neo4j:")
    print("\nNode Types:")
    print("  - Visitor_this_year: Current year visitors")
    print("  - Visitor_last_year_bva: Previous year BVA visitors")
    print("  - Visitor_last_year_lva: Previous year LVA visitors")
    print("  - Sessions_this_year: Current year sessions")
    print("  - Sessions_past_year: Previous year sessions")
    print("  - Stream: Session categories/streams")
    
    print("\nRelationship Types:")
    print("  - HAS_STREAM: Sessions to their stream categories")
    print("  - job_to_stream: Visitors to relevant streams based on job roles")
    print("  - specialization_to_stream: Visitors to streams based on specializations")
    print("  - Same_Visitor: Links visitors across years")
    print("  - attended_session: Visitors to sessions they attended")
    
    print("\nExample Queries:")
    print("1. Count visitor nodes:")
    print("   MATCH (v:Visitor_this_year) RETURN count(v)")
    
    print("\n2. Find sessions in a specific stream:")
    print("   MATCH (s:Sessions_this_year)-[:HAS_STREAM]->(st:Stream {stream: 'cardiology'})")
    print("   RETURN s.title, s.date, s.start_time")
    
    print("\n3. Find visitors with job-to-stream relationships:")
    print("   MATCH (v:Visitor_this_year)-[:job_to_stream]->(s:Stream)")
    print("   RETURN v.job_role, s.stream, count(*) as connections")
    print("   ORDER BY connections DESC LIMIT 10")
    
    print("\n4. Find recommended sessions for a visitor:")
    print("   MATCH (v:Visitor_this_year {BadgeId: 'your_badge_id'})-[:job_to_stream]->(s:Stream)")
    print("   MATCH (session:Sessions_this_year)-[:HAS_STREAM]->(s)")
    print("   RETURN session.title, session.date, s.stream")
    
    print("=" * 60)