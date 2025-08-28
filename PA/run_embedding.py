#!/usr/bin/env python3
"""
Script to directly run only the session embedding processor.
python run_embedding.py --recreate-all
"""


import argparse
import sys
from utils.logging_utils import setup_logging
from utils.config_utils import load_config
from session_embedding_processor import SessionEmbeddingProcessor

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process session embeddings in Neo4j.")
    parser.add_argument(
        "--recreate-all",
        action="store_true",
        help="Recreate all embeddings, even if they already exist in Neo4j",
    )
    args = parser.parse_args()

    create_only_new = not args.recreate_all

    # Set up logging first
    logger = setup_logging(log_file="logs/embedding_processing.log")

    try:
        # Load the configuration
        config = load_config("config/config.yaml")

        logger.info("Starting session embedding processing")

        # Create and run the embedding processor
        embedding_processor = SessionEmbeddingProcessor(config)
        result = embedding_processor.process(create_only_new=create_only_new)

        if result:
            logger.info("Session embedding processing completed successfully")

            # Print summary statistics
            stats = embedding_processor.statistics
            logger.info(
                f"Total sessions processed: {stats['total_sessions_processed']}"
            )
            logger.info(
                f"Sessions with embeddings: {stats['sessions_with_embeddings']}"
            )
            logger.info(
                f"Sessions with stream descriptions: {stats['sessions_with_stream_descriptions']}"
            )
            logger.info(
                f"Sessions this year: {stats['sessions_by_type']['sessions_this_year']}"
            )
            logger.info(
                f"Sessions past year: {stats['sessions_by_type']['sessions_past_year']}"
            )

            print("\nEmbedding Processing Summary:")
            print(f"Total sessions processed: {stats['total_sessions_processed']}")
            print(f"Sessions with embeddings: {stats['sessions_with_embeddings']}")
            print(
                f"Sessions with stream descriptions: {stats['sessions_with_stream_descriptions']}"
            )
            print(
                f"Sessions this year: {stats['sessions_by_type']['sessions_this_year']}"
            )
            print(
                f"Sessions past year: {stats['sessions_by_type']['sessions_past_year']}"
            )
            sys.exit(0)  # Success exit code
        else:
            logger.error("Session embedding processing failed")
            print("Embedding processing failed. See logs for details.")
            sys.exit(1)  # Error exit code

    except Exception as e:
        logger.error(f"Error in embedding processing: {e}", exc_info=True)
        print(f"Error: {e}")
        print("See logs/embedding_processing.log for details")
        sys.exit(1)  # Error exit code
