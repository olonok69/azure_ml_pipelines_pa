#!/usr/bin/env python3
"""
Script to directly run only the session recommendation processor.
"""

import argparse
import sys
import os
from utils.logging_utils import setup_logging
from utils.config_utils import load_config
from recommendation_processor import RecommendationProcessor

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process session recommendations for visitors."
    )
    parser.add_argument(
        "--visitor-data",
        type=str,
        help="Path to CSV file with visitor data",
    )
    parser.add_argument(
        "--specific-badges",
        type=str,
        help="Comma-separated list of specific badge IDs to process",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.3,
        help="Minimum similarity score (0.0-1.0)",
    )
    parser.add_argument(
        "--max-recommendations",
        type=int,
        default=10,
        help="Maximum number of recommendations per visitor",
    )
    args = parser.parse_args()

    # Set up logging first
    logger = setup_logging(log_file="logs/recommendation_processing.log")

    try:
        # Load the configuration
        config = load_config("config/config.yaml")

        # Override config with command line arguments if provided
        if args.min_score is not None:
            if "recommendations" not in config:
                config["recommendations"] = {}
            config["recommendations"]["min_score"] = args.min_score

        if args.max_recommendations is not None:
            if "recommendations" not in config:
                config["recommendations"] = {}
            config["recommendations"]["max_recommendations"] = args.max_recommendations

        logger.info("Starting session recommendation processing")

        # Process specific badges if provided
        specific_badges = None
        if args.specific_badges:
            specific_badges = [
                badge.strip() for badge in args.specific_badges.split(",")
            ]
            logger.info(f"Processing specific badges: {specific_badges}")

        # Create and run the recommendation processor
        recommendation_processor = RecommendationProcessor(config)
        result = recommendation_processor.process(
            visitor_data_path=args.visitor_data, specific_badges=specific_badges
        )

        if result:
            logger.info("Session recommendation processing completed successfully")

            # Print summary statistics
            stats = recommendation_processor.statistics
            logger.info(f"Visitors processed: {stats['visitors_processed']}")
            logger.info(
                f"Visitors with recommendations: {stats['visitors_with_recommendations']}"
            )
            logger.info(
                f"Total raw recommendations: {stats['total_recommendations_generated']}"
            )
            logger.info(
                f"Total filtered recommendations: {stats['total_filtered_recommendations']}"
            )

            print("\nRecommendation Processing Summary:")
            print(f"Total visitors processed: {stats['visitors_processed']}")
            print(
                f"Visitors with recommendations: {stats['visitors_with_recommendations']}"
            )
            print(
                f"Visitors without recommendations: {stats['visitors_without_recommendations']}"
            )
            print(
                f"Total raw recommendations: {stats['total_recommendations_generated']}"
            )
            print(
                f"Total filtered recommendations: {stats['total_filtered_recommendations']}"
            )

            avg_recommendations = 0
            if stats["visitors_with_recommendations"] > 0:
                avg_recommendations = (
                    stats["total_filtered_recommendations"]
                    / stats["visitors_with_recommendations"]
                )
            print(f"Average recommendations per visitor: {avg_recommendations:.1f}")

            print(f"Processing time: {stats['processing_time']:.2f} seconds")

            # Print output location
            print(
                f"\nRecommendation files saved to: {os.path.abspath(os.path.join(config['output_dir'], 'output'))}"
            )
            sys.exit(0)  # Success exit code
        else:
            logger.error("Session recommendation processing failed")
            print("Recommendation processing failed. See logs for details.")
            sys.exit(1)  # Error exit code

    except Exception as e:
        logger.error(f"Error in recommendation processing: {e}", exc_info=True)
        print(f"Error: {e}")
        print("See logs/recommendation_processing.log for details")
        sys.exit(1)  # Error exit code
