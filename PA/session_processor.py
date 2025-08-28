#!/usr/bin/env python3
"""
Generic Session Data Processor

This processor handles session data for various events, extracting unique streams,
generating descriptions, and preparing data for analytics.

FIXED: Matches production old_session_processor.py functionality exactly.
- Correct key_text handling (create from title, not expect it in input)
- NO stream normalization - keeps original case
- Exact title filtering using .isin() not .contains()
"""

import os
import json
import string
import logging
import pandas as pd
import functools
from typing import Dict, List, Set, Any
from dotenv import load_dotenv, dotenv_values
from langchain_openai import ChatOpenAI, AzureChatOpenAI


# Configure logger
logger = logging.getLogger(__name__)


# Optional decorator for logging function call
def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        logger.debug(f"Exiting {func.__name__}")
        return result

    return wrapper


class SessionProcessor:
    """Process session data for event analytics."""

    def __init__(self, config: Dict):
        """
        Initialize the processor with configuration.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.output_dir = config.get("output_dir", "output")

        # Initialize the use_cached_descriptions parameter from config (default is True)
        self.use_cached_descriptions = config.get("stream_processing", {}).get(
            "use_cached_descriptions", True
        )
        self.cache_file_path = os.path.join(
            self.output_dir, "output", "streams_cache.json"
        )

        # Initialize logger first, before any other operations that use it
        self.logger = logging.getLogger(__name__)

        # Load environment variables for API access
        self.env_file = config.get("env_file", "keys/.env")
        self.load_env_variables()

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "output"), exist_ok=True)

        self.logger.info(
            f"Initialized SessionProcessor with output to {self.output_dir}"
        )

        # Load map_vets and titles_to_remove from config
        self.map_vets = self.config.get("map_vets", {})
        self.titles_to_remove = self.config.get("titles_to_remove", [])

    def load_env_variables(self) -> None:
        """Load environment variables for API access."""
        try:
            status = load_dotenv(self.env_file)
            self.env_config = dotenv_values(self.env_file)

            # Check for either OpenAI or Azure OpenAI API credentials
            if not status:
                self.logger.warning("Failed to load environment variables")
                raise ValueError("Failed to load environment variables")

            # Check for OpenAI credentials
            if "OPENAI_API_KEY" not in self.env_config and (
                "AZURE_API_KEY" not in self.env_config
                or "AZURE_ENDPOINT" not in self.env_config
                or "AZURE_DEPLOYMENT" not in self.env_config
                or "AZURE_API_VERSION" not in self.env_config
            ):
                self.logger.warning("API credentials missing")
                raise ValueError("API credentials not found in environment variables")

            self.logger.info("Successfully loaded environment variables")
        except Exception as e:
            self.logger.error(
                f"Error loading environment variables: {e}", exc_info=True
            )
            raise

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Remove punctuation and spaces, and convert to lowercase.

        Args:
            text: Text to be cleaned

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        return "".join(char for char in text if char.isalnum()).lower()

    def load_session_data(self) -> None:
        """Load all session data files based on configuration."""
        try:
            # Load session data
            session_this_path = self.config["session_files"]["session_this"]
            session_past_path_bva = self.config["session_files"]["session_past_bva"]
            session_past_path_lva = self.config["session_files"]["session_past_lva"]

            # Read session data
            self.session_this = pd.read_csv(session_this_path)
            self.session_past_bva = pd.read_csv(session_past_path_bva)
            self.session_past_lva = pd.read_csv(session_past_path_lva)

            self.logger.info(
                f"Loaded session data: {len(self.session_this)} records this year, "
                f"{len(self.session_past_bva)} BVA records last year, "
                f"{len(self.session_past_lva)} LVA records last year"
            )

        except Exception as e:
            self.logger.error(f"Error loading session data: {e}", exc_info=True)
            raise

    def filter_session_data(self) -> None:
        """Filter and clean session data - MATCHES PRODUCTION EXACTLY."""
        # Convert titles_to_remove to lowercase for case-insensitive comparison
        titles_to_remove_lower = [title.lower() for title in self.titles_to_remove]
        try:
            # Apply filters for each dataset
            self.session_this_filtered = self.session_this.copy()
            self.session_last_filtered_bva = self.session_past_bva.copy()
            self.session_last_filtered_lva = self.session_past_lva.copy()

            # Apply title filters
            # Filter out rows where the title EXACTLY matches any in titles_to_remove (case-insensitive)
            self.session_this_filtered = self.session_this_filtered[
                ~self.session_this_filtered["title"]
                .str.lower()
                .isin(titles_to_remove_lower)
            ]

            self.session_last_filtered_bva = self.session_last_filtered_bva[
                ~self.session_last_filtered_bva["title"]
                .str.lower()
                .isin(titles_to_remove_lower)
            ]

            self.session_last_filtered_lva = self.session_last_filtered_lva[
                ~self.session_last_filtered_lva["title"]
                .str.lower()
                .isin(titles_to_remove_lower)
            ]

            # Create text keys for matching - FROM TITLE FIELD
            self.session_this_filtered["key_text"] = self.session_this_filtered[
                "title"
            ].apply(self.clean_text)
            self.session_last_filtered_bva["key_text"] = self.session_last_filtered_bva[
                "title"
            ].apply(self.clean_text)
            self.session_last_filtered_lva["key_text"] = self.session_last_filtered_lva[
                "title"
            ].apply(self.clean_text)

            # Remove duplicates based on key_text
            self.session_this_filtered = self.session_this_filtered.drop_duplicates(
                subset=["key_text"]
            )
            self.session_last_filtered_bva = (
                self.session_last_filtered_bva.drop_duplicates(subset=["key_text"])
            )
            self.session_last_filtered_lva = (
                self.session_last_filtered_lva.drop_duplicates(subset=["key_text"])
            )

            self.logger.info(
                f"Filtered session data: {len(self.session_this_filtered)} records this year, "
                f"{len(self.session_last_filtered_bva)} BVA records last year, "
                f"{len(self.session_last_filtered_lva)} LVA records last year"
            )

        except Exception as e:
            self.logger.error(f"Error filtering session data: {e}", exc_info=True)
            raise

    def select_relevant_columns(self) -> None:
        """Select only relevant columns from session data."""
        try:
            # Define the columns we want to keep - maintain consistent order
            cols_to_keep = [
                "session_id",
                "date",
                "start_time", 
                "end_time",
                "theatre__name",
                "title",
                "stream",
                "synopsis_stripped",
                "sponsored_session",
                "sponsored_by",
                "key_text",  # This was created in filter_session_data
            ]

            # Filter columns for each dataset
            self.session_this_filtered_valid_cols = self.session_this_filtered[
                cols_to_keep
            ].copy()
            self.session_last_filtered_valid_cols_bva = self.session_last_filtered_bva[
                cols_to_keep
            ].copy()
            self.session_last_filtered_valid_cols_lva = self.session_last_filtered_lva[
                cols_to_keep
            ].copy()

            # Fill NaN values with empty strings
            self.session_this_filtered_valid_cols = (
                self.session_this_filtered_valid_cols.fillna("")
            )
            self.session_last_filtered_valid_cols_bva = (
                self.session_last_filtered_valid_cols_bva.fillna("")
            )
            self.session_last_filtered_valid_cols_lva = (
                self.session_last_filtered_valid_cols_lva.fillna("")
            )

            self.logger.info("Selected relevant columns and cleaned data")

        except Exception as e:
            self.logger.error(f"Error selecting relevant columns: {e}", exc_info=True)
            raise

    def extract_unique_streams(self) -> None:
        """Extract unique stream values from all session datasets."""
        try:
            # Collect all stream values
            all_streams = set()

            # Extract streams from each dataset
            for df in [
                self.session_this_filtered_valid_cols,
                self.session_last_filtered_valid_cols_bva,
                self.session_last_filtered_valid_cols_lva,
            ]:
                for stream_cell in df["stream"]:
                    if pd.notna(stream_cell) and stream_cell:
                        # Split by ';' and clean each stream - NO LOWERCASE CONVERSION
                        streams = [s.strip() for s in str(stream_cell).split(";")]
                        all_streams.update(streams)

            # Remove empty strings and sort
            self.unique_streams = sorted([s for s in all_streams if s])
            
            # Also create 'streams' attribute for backward compatibility
            self.streams = self.unique_streams

            self.logger.info(f"Extracted {len(self.unique_streams)} unique streams")

        except Exception as e:
            self.logger.error(f"Error extracting unique streams: {e}", exc_info=True)
            raise

    def setup_language_model(self) -> None:
        """Set up the language model for generating stream descriptions."""
        try:
            # Try to load cached descriptions first
            if self.use_cached_descriptions and os.path.exists(self.cache_file_path):
                with open(self.cache_file_path, "r") as f:
                    self.streams_catalog = json.load(f)
                    self.logger.info(
                        f"Loaded cached stream descriptions from {self.cache_file_path}"
                    )
                    return

            # Check for Azure OpenAI credentials first
            if "AZURE_API_KEY" in self.env_config:
                from langchain_openai import AzureChatOpenAI
                self.llm = AzureChatOpenAI(
                    azure_endpoint=self.env_config.get("AZURE_ENDPOINT"),
                    openai_api_version=self.env_config.get("AZURE_API_VERSION"),
                    deployment_name=self.env_config.get("AZURE_DEPLOYMENT", "gpt-4o-mini"),
                    openai_api_key=self.env_config["AZURE_API_KEY"],
                    temperature=0.7,
                )
                self.logger.info("Initialized Azure OpenAI language model")
            elif "OPENAI_API_KEY" in self.env_config:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.7,
                    openai_api_key=self.env_config["OPENAI_API_KEY"],
                )
                self.logger.info("Initialized OpenAI language model")
            else:
                self.logger.warning(
                    "No API keys available - stream descriptions will be skipped"
                )
                self.llm = None

        except Exception as e:
            self.logger.error(f"Error setting up language model: {e}", exc_info=True)
            self.llm = None

    def save_cached_descriptions(self) -> None:
        """Save generated stream descriptions to cache file."""
        try:
            if hasattr(self, "streams_catalog") and self.streams_catalog:
                with open(self.cache_file_path, "w") as f:
                    json.dump(self.streams_catalog, f, indent=2)
                self.logger.info(f"Saved stream descriptions to cache: {self.cache_file_path}")
        except Exception as e:
            self.logger.error(f"Error saving cached descriptions: {e}", exc_info=True)

    def generate_stream_descriptions(self) -> None:
        """Generate descriptions for each unique stream using LLM."""
        # Only generate if we don't have cached descriptions
        if self.use_cached_descriptions and hasattr(self, 'streams_catalog') and self.streams_catalog:
            self.logger.info("Using existing cached stream descriptions")
            return

        # If no LLM is set up, skip generation
        if not hasattr(self, "llm") or self.llm is None:
            self.logger.info("No language model configured - creating empty catalog")
            self.streams_catalog = {}
            return

        try:
            self.logger.info("Generating new stream descriptions using language model")
            self.streams_catalog = {}
            
            for stream in self.unique_streams:
                prompt = f"""
                Provide a brief description (2-3 sentences) for the following conference stream: "{stream}"
                
                Focus on what topics and sessions would typically be covered in this stream.
                Keep the description professional and informative.
                """

                response = self.llm.invoke(prompt)
                
                # Store as simple dictionary (old format) - NO LOWERCASE
                self.streams_catalog[stream] = response.content.strip()
                self.logger.debug(f"Generated description for stream: {stream}")

            self.logger.info(f"Generated descriptions for {len(self.streams_catalog)} streams")
            
            # Save the newly generated descriptions to cache
            self.save_cached_descriptions()

        except Exception as e:
            self.logger.error(f"Error generating stream catalog: {e}", exc_info=True)
            self.streams_catalog = {}
            raise

    def generate_stream_catalog(self) -> None:
        """Generate or load the stream catalog."""
        # This method is kept for compatibility but most logic is in generate_stream_descriptions
        if not hasattr(self, 'streams_catalog'):
            self.streams_catalog = {}
        
        self.logger.info(f"Stream catalog ready with {len(self.streams_catalog)} entries")

    def save_streams_catalog(self) -> None:
        """Save the streams catalog to JSON file."""
        try:
            output_path = os.path.join(self.output_dir, "output", "streams.json")

            with open(output_path, "w") as json_file:
                json.dump(self.streams_catalog, json_file, indent=4)

            self.logger.info(f"Saved stream catalog to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving stream catalog: {e}", exc_info=True)
            raise

    @staticmethod
    def find_short_labels(input_set: Set[str]) -> List[str]:
        """
        Finds a list of labels in a set that have 5 characters or less.

        Args:
          input_set: A set of strings (labels).

        Returns:
          A list containing the labels from the input set with a length of 5 or less.
        """
        short_labels = [
            label for label in input_set if isinstance(label, str) and len(label) <= 5
        ]
        return short_labels

    def expand_sponsor_abbreviations(self) -> None:
        """Expand sponsor abbreviations to full names."""
        try:
            # Get all unique sponsor values
            list_bva_this = set(
                list(self.session_this_filtered_valid_cols.sponsored_by.unique())
            )
            list_lva_last = set(
                list(self.session_last_filtered_valid_cols_lva.sponsored_by.unique())
            )
            list_bva_last = set(
                list(self.session_last_filtered_valid_cols_bva.sponsored_by.unique())
            )

            # Combine all sponsor values
            full_list_sponsors = list_bva_this.union(list_lva_last, list_bva_last)

            # Find abbreviations (short labels)
            list_abbreviations = set(self.find_short_labels(full_list_sponsors))

            # Check for any missing abbreviations in our mapping
            map_keys = set(list(self.map_vets.keys()))
            missing_abbrevs = list_abbreviations.difference(map_keys)

            if missing_abbrevs:
                self.logger.warning(f"Found unmapped abbreviations: {missing_abbrevs}")

            # Replace abbreviations with full names using pandas replace method
            self.session_last_filtered_valid_cols_lva["sponsored_by"] = (
                self.session_last_filtered_valid_cols_lva["sponsored_by"].replace(
                    self.map_vets
                )
            )
            self.session_last_filtered_valid_cols_bva["sponsored_by"] = (
                self.session_last_filtered_valid_cols_bva["sponsored_by"].replace(
                    self.map_vets
                )
            )
            self.session_this_filtered_valid_cols["sponsored_by"] = (
                self.session_this_filtered_valid_cols["sponsored_by"].replace(
                    self.map_vets
                )
            )

            self.logger.info("Expanded sponsor abbreviations to full names")

        except Exception as e:
            self.logger.error(
                f"Error expanding sponsor abbreviations: {e}", exc_info=True
            )
            raise

    def save_processed_session_data(self) -> None:
        """Save processed session data to CSV files."""
        try:
            # Define output paths
            bva_last_output_path = os.path.join(
                self.output_dir, "output", "session_last_filtered_valid_cols_bva.csv"
            )
            lva_last_output_path = os.path.join(
                self.output_dir, "output", "session_last_filtered_valid_cols_lva.csv"
            )
            this_output_path = os.path.join(
                self.output_dir, "output", "session_this_filtered_valid_cols.csv"
            )

            # Save to CSV - NO NORMALIZATION
            self.session_last_filtered_valid_cols_bva.to_csv(
                bva_last_output_path, index=False
            )
            self.session_last_filtered_valid_cols_lva.to_csv(
                lva_last_output_path, index=False
            )
            self.session_this_filtered_valid_cols.to_csv(this_output_path, index=False)

            self.logger.info(
                f"Saved processed session data to {self.output_dir}/output/"
            )
            self.logger.info(
                f"BVA last year sessions: {len(self.session_last_filtered_valid_cols_bva)} records"
            )
            self.logger.info(
                f"LVA last year sessions: {len(self.session_last_filtered_valid_cols_lva)} records"
            )
            self.logger.info(
                f"This year sessions: {len(self.session_this_filtered_valid_cols)} records"
            )

        except Exception as e:
            self.logger.error(
                f"Error saving processed session data: {e}", exc_info=True
            )
            raise

    def process(self) -> None:
        """Execute the full session data processing workflow."""
        self.logger.info("Starting session data processing workflow")

        self.load_session_data()
        self.filter_session_data()
        self.select_relevant_columns()
        self.extract_unique_streams()
        self.setup_language_model()
        self.generate_stream_descriptions()
        self.generate_stream_catalog()
        self.save_streams_catalog()
        self.expand_sponsor_abbreviations()
        self.save_processed_session_data()

        self.logger.info("Session data processing workflow completed successfully")


def main():
    """Main function for testing the session processor."""
    import sys
    sys.path.insert(0, os.getcwd())
    
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging

    # Setup logging
    logger = setup_logging(log_file="logs/session_processor.log")
    
    try:
        # Load configuration
        config = load_config("config/config_vet.yaml")
        
        # Create processor and run
        processor = SessionProcessor(config)
        processor.process()
        
        print("Session processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Session processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()