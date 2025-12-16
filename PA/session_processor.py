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
import re
import json
import logging
import shutil
import functools
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Any, Optional

import pandas as pd
from dotenv import load_dotenv, dotenv_values
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


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
        self.session_output_config = self.config.get("session_output_files", {})
        self.session_processed_files = self.session_output_config.get(
            "processed_sessions", {}
        )

        self.stream_processing_config = self.config.get("stream_processing", {})
        self.create_missing_streams = self.stream_processing_config.get(
            "create_missing_streams", False
        )
        self.use_enhanced_streams_catalog = self.stream_processing_config.get(
            "use_enhanced_streams_catalog", False
        )

        self.session_file_paths: Dict[str, str] = {}
        self.backfill_metrics: Dict[str, int] = {
            "files_evaluated": 0,
            "files_modified": 0,
            "total_missing_streams": 0,
            "sessions_backfilled": 0,
            "sessions_skipped_empty_synopsis": 0,
            "sessions_skipped_no_candidates": 0,
            "sessions_failed_llm": 0,
        }
        self._streams_by_theatre: Optional[Dict[str, Set[str]]] = None

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

            # Deterministic sort before de-dup so we consistently keep the earliest scheduled session
            def stable_sort(df: pd.DataFrame) -> pd.DataFrame:
                sort_cols = [c for c in ["date", "start_time", "session_id"] if c in df.columns]
                if sort_cols:
                    return df.sort_values(by=sort_cols, ascending=True, kind="stable")
                return df

            self.session_this_filtered = stable_sort(self.session_this_filtered)
            self.session_last_filtered_bva = stable_sort(self.session_last_filtered_bva)
            self.session_last_filtered_lva = stable_sort(self.session_last_filtered_lva)

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

            # Upstream data-quality gate: drop and quarantine placeholder rows (e.g., blank titles)
            def drop_and_quarantine(df: pd.DataFrame, name: str) -> pd.DataFrame:
                required_text_fields = ["title", "synopsis_stripped", "theatre__name", "key_text", "stream"]
                # Identify placeholder rows: all core text fields blank OR title blank
                is_blank = df[required_text_fields].apply(lambda s: s.astype(str).str.strip() == "").all(axis=1)
                is_title_blank = df["title"].astype(str).str.strip() == ""
                mask_quarantine = is_blank | is_title_blank
                quarantined = df[mask_quarantine].copy()
                cleaned = df[~mask_quarantine].copy()

                if len(quarantined) > 0:
                    q_path = os.path.join(self.output_dir, "output", f"quarantined_sessions_{name}.csv")
                    try:
                        quarantined.to_csv(q_path, index=False)
                        self.logger.warning(
                            f"Quarantined {len(quarantined)} placeholder sessions from {name} → {q_path}"
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to write quarantine file for {name}: {e}")
                return cleaned

            self.session_this_filtered_valid_cols = drop_and_quarantine(
                self.session_this_filtered_valid_cols, "this_year"
            )
            self.session_last_filtered_valid_cols_bva = drop_and_quarantine(
                self.session_last_filtered_valid_cols_bva, "past_year_bva"
            )
            self.session_last_filtered_valid_cols_lva = drop_and_quarantine(
                self.session_last_filtered_valid_cols_lva, "past_year_lva"
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
            streams_filename = self.session_output_config.get(
                "streams_catalog", "streams.json"
            )
            output_path = os.path.join(self.output_dir, "output", streams_filename)

            with open(output_path, "w") as json_file:
                json.dump(self.streams_catalog, json_file, indent=4)

            self.logger.info(f"Saved stream catalog to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving stream catalog: {e}", exc_info=True)
            raise

    def _ensure_language_model(self) -> bool:
        """Ensure an LLM instance is available for downstream classification."""
        if hasattr(self, "llm") and self.llm:
            return True

        try:
            if "AZURE_API_KEY" in self.env_config:
                self.llm = AzureChatOpenAI(
                    azure_endpoint=self.env_config.get("AZURE_ENDPOINT"),
                    openai_api_version=self.env_config.get("AZURE_API_VERSION"),
                    deployment_name=self.env_config.get(
                        "AZURE_DEPLOYMENT", "gpt-4o-mini"
                    ),
                    openai_api_key=self.env_config["AZURE_API_KEY"],
                    temperature=0.3,
                )
                self.logger.info("Initialized Azure OpenAI language model for stream backfill")
                return True

            if "OPENAI_API_KEY" in self.env_config:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    openai_api_key=self.env_config["OPENAI_API_KEY"],
                )
                self.logger.info("Initialized OpenAI language model for stream backfill")
                return True

            self.logger.warning(
                "Language model credentials not available; stream backfill will be skipped"
            )
            self.llm = None
            return False

        except Exception as e:
            self.logger.error(
                f"Failed to initialize language model for stream backfill: {e}",
                exc_info=True,
            )
            self.llm = None
            return False

    def _load_stream_catalog_for_backfill(self) -> Dict[str, str]:
        """Load the stream catalog used for classifying missing streams."""
        if hasattr(self, "_backfill_stream_catalog"):
            return getattr(self, "_backfill_stream_catalog")

        streams_filename = self.session_output_config.get(
            "streams_catalog", "streams.json"
        )
        output_root = os.path.join(self.output_dir, "output")

        candidate_paths: List[str] = []
        if self.use_enhanced_streams_catalog:
            base, ext = os.path.splitext(streams_filename)
            enhanced_name = f"{base}_enhanced{ext}" if ext else f"{streams_filename}_enhanced"
            candidate_paths.append(os.path.join(output_root, enhanced_name))
            candidate_paths.append(os.path.join(output_root, "streams_enhanced.json"))

        candidate_paths.append(os.path.join(output_root, streams_filename))

        catalog: Dict[str, str] = {}
        for path in candidate_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    catalog = data
                    self.logger.info(
                        f"Loaded stream catalog for backfill from {path}"
                    )
                    break
                self.logger.warning(
                    f"Stream catalog at {path} is not a dictionary. Skipping"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to load stream catalog from {path}: {e}", exc_info=True
                )

        if not catalog and getattr(self, "streams_catalog", None):
            catalog = self.streams_catalog
            self.logger.info("Using in-memory streams catalog for backfill")

        if not catalog:
            self.logger.warning(
                "Stream catalog unavailable; missing stream backfill will be skipped"
            )

        setattr(self, "_backfill_stream_catalog", catalog)
        return catalog

    @staticmethod
    def _is_stream_missing(value: Any) -> bool:
        """Return True when the provided stream value should be considered missing."""
        if value is None:
            return True
        if isinstance(value, float) and pd.isna(value):
            return True
        text = str(value).strip()
        return text == "" or text.lower() in {"nan", "none", "null", "na"}

    def _missing_stream_mask(self, df: pd.DataFrame) -> pd.Series:
        """Build a boolean mask for rows with missing stream assignments."""
        return df["stream"].apply(self._is_stream_missing)

    def _iter_session_dataframes(self) -> List[pd.DataFrame]:
        """Return processed session DataFrames that can be inspected for streams."""
        frames: List[pd.DataFrame] = []
        for attr in [
            "session_this_filtered_valid_cols",
            "session_last_filtered_valid_cols_bva",
            "session_last_filtered_valid_cols_lva",
        ]:
            df = getattr(self, attr, None)
            if isinstance(df, pd.DataFrame):
                frames.append(df)
        return frames

    def _build_streams_by_theatre(self) -> Dict[str, Set[str]]:
        """Create a lookup of theatre name to known stream assignments."""
        theatre_map: Dict[str, Set[str]] = defaultdict(set)
        for df in self._iter_session_dataframes():
            if df.empty or "theatre__name" not in df or "stream" not in df:
                continue
            for _, row in df.iterrows():
                theatre = str(row.get("theatre__name", "")).strip()
                if not theatre:
                    continue
                stream_value = row.get("stream", "")
                if self._is_stream_missing(stream_value):
                    continue
                for stream_name in str(stream_value).split(";"):
                    cleaned = stream_name.strip()
                    if cleaned:
                        theatre_map[theatre.lower()].add(cleaned)
        return theatre_map

    def _get_streams_for_theatre(self, theatre_name: str) -> Set[str]:
        """Return known streams associated with a theatre."""
        if not theatre_name:
            return set()
        if self._streams_by_theatre is None:
            self._streams_by_theatre = self._build_streams_by_theatre()
        return set(
            self._streams_by_theatre.get(theatre_name.strip().lower(), set())
        )

    def _register_streams_for_theatre(self, theatre_name: str, streams: List[str]) -> None:
        """Update the theatre to streams lookup when new assignments are created."""
        if not theatre_name or not streams:
            return
        if self._streams_by_theatre is None:
            self._streams_by_theatre = self._build_streams_by_theatre()
        key = theatre_name.strip().lower()
        theatre_set = self._streams_by_theatre.setdefault(key, set())
        theatre_set.update(streams)

    def _build_classification_messages(
        self,
        stream_candidates: List[str],
        catalog: Dict[str, str],
        title: str,
        synopsis: str,
    ) -> List[Any]:
        """Construct system and human messages for stream classification."""
        system_content = (
            "You classify conference sessions into existing stream categories. "
            "Choose between 1 and 3 stream names from the provided list that best match the session. "
            "Respond with the stream names separated by semicolons on a single line using the exact names provided. "
            "Do not add any additional text."
        )

        stream_lines = [f"{name}: {catalog.get(name, '')}" for name in stream_candidates]
        human_content = (
            "Streams:\n"
            + "\n".join(stream_lines)
            + "\n\n"
            + f"Title: {title}\n"
            + f"Synopsis: {synopsis}"
        )

        return [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

    def _parse_stream_response(
        self, response_text: str, available_streams: List[str]
    ) -> List[str]:
        """Parse the LLM response into canonical stream names."""
        if not response_text:
            return []

        canonical_map = {name.lower(): name for name in available_streams}
        parsed: List[str] = []

        for fragment in re.split(r"[;\n]+", response_text):
            candidate = fragment.strip().strip("-*•\"')")
            if not candidate:
                continue

            lower_candidate = candidate.lower()
            match = canonical_map.get(lower_candidate)

            if not match:
                simplified = re.sub(r"[^a-z0-9 &/+]", "", lower_candidate)
                for stream_name in available_streams:
                    if re.sub(r"[^a-z0-9 &/+]", "", stream_name.lower()) == simplified:
                        match = stream_name
                        break

            if match and match not in parsed:
                parsed.append(match)
            if len(parsed) == 3:
                break

        return parsed

    def _classify_streams_for_row(
        self,
        row: pd.Series,
        stream_catalog: Dict[str, str],
        candidate_streams: List[str],
    ) -> List[str]:
        """Call the language model to classify a session into streams."""
        title = str(row.get("title", "")).strip()
        synopsis = str(row.get("synopsis_stripped", "")).strip()

        messages = self._build_classification_messages(
            candidate_streams, stream_catalog, title, synopsis
        )

        try:
            response = self.llm.invoke(messages)
            raw_output: Optional[str] = None
            if hasattr(response, "content"):
                raw_output = response.content
            elif isinstance(response, str):
                raw_output = response
            elif hasattr(response, "text"):
                raw_output = response.text

            if raw_output is None:
                self.logger.warning(
                    f"Empty response received when classifying session '{title}'"
                )
                return []

            return self._parse_stream_response(raw_output.strip(), candidate_streams)

        except Exception as e:
            self.logger.error(
                f"Language model classification failed for session '{title}': {e}",
                exc_info=True,
            )
            return []

    def _backup_file(self, file_path: str) -> Optional[str]:
        """Create a timestamped backup of the supplied file."""
        if not file_path or not os.path.exists(file_path):
            self.logger.warning(f"Cannot back up missing file: {file_path}")
            return None

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = f"{file_path}.{timestamp}.bak"

        try:
            shutil.copy2(file_path, backup_path)
            self.logger.info(f"Created backup {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(
                f"Failed to create backup for {file_path}: {e}", exc_info=True
            )
            return None

    def backfill_missing_streams(self) -> None:
        """Backfill missing stream values using the language model."""
        if not self.create_missing_streams:
            self.logger.info(
                "Stream backfill disabled via configuration (stream_processing.create_missing_streams)"
            )
            return

        stream_catalog = self._load_stream_catalog_for_backfill()
        if not stream_catalog:
            return

        if not self._ensure_language_model():
            return

        self._streams_by_theatre = self._build_streams_by_theatre()

        dataframes = [
            ("this_year", getattr(self, "session_this_filtered_valid_cols", None)),
            (
                "last_year_main",
                getattr(self, "session_last_filtered_valid_cols_bva", None),
            ),
            (
                "last_year_secondary",
                getattr(self, "session_last_filtered_valid_cols_lva", None),
            ),
        ]

        for label, df in dataframes:
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            output_path = self.session_file_paths.get(label)
            self.backfill_metrics["files_evaluated"] += 1

            if not output_path:
                self.logger.warning(
                    f"No output path recorded for {label} sessions; skipping stream backfill"
                )
                continue

            missing_mask = self._missing_stream_mask(df)
            missing_indices = df.index[missing_mask].tolist()
            missing_count = len(missing_indices)

            if missing_count == 0:
                self.logger.info(f"No missing streams detected in {label} sessions")
                continue

            self.backfill_metrics["total_missing_streams"] += missing_count
            self._backup_file(output_path)

            backfilled_count = 0

            for idx in missing_indices:
                row = df.loc[idx]
                synopsis = str(row.get("synopsis_stripped", "")).strip()
                if not synopsis:
                    self.backfill_metrics["sessions_skipped_empty_synopsis"] += 1
                    continue

                theatre_name = str(row.get("theatre__name", "")).strip()
                theatre_streams = list(self._get_streams_for_theatre(theatre_name))

                candidate_streams = (
                    sorted(theatre_streams)
                    if theatre_streams
                    else sorted(stream_catalog.keys())
                )

                if not candidate_streams:
                    self.backfill_metrics["sessions_skipped_no_candidates"] += 1
                    continue

                # Limit prompt size to keep requests efficient
                if len(candidate_streams) > 60:
                    candidate_streams = candidate_streams[:60]

                classifications = self._classify_streams_for_row(
                    row, stream_catalog, candidate_streams
                )

                if not classifications:
                    self.backfill_metrics["sessions_failed_llm"] += 1
                    continue

                normalized_streams = "; ".join(classifications)
                df.at[idx, "stream"] = normalized_streams
                backfilled_count += 1
                self.backfill_metrics["sessions_backfilled"] += 1
                self._register_streams_for_theatre(theatre_name, classifications)

            if backfilled_count > 0:
                df.to_csv(output_path, index=False)
                self.backfill_metrics["files_modified"] += 1
                self.logger.info(
                    f"Backfilled {backfilled_count} sessions with missing streams in {label} ({output_path})"
                )
            else:
                self.logger.info(
                    f"No sessions backfilled in {label}; all candidates were skipped"
                )

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
            event_config = self.config.get("event", {})
            main_event_name = event_config.get("main_event_name", "bva")
            secondary_event_name = event_config.get("secondary_event_name", "lva")

            this_year_filename = self.session_processed_files.get(
                "this_year", "session_this_filtered_valid_cols.csv"
            )
            last_year_main_filename = self.session_processed_files.get(
                "last_year_main",
                f"session_last_filtered_valid_cols_{main_event_name}.csv",
            )
            last_year_secondary_filename = self.session_processed_files.get(
                "last_year_secondary",
                f"session_last_filtered_valid_cols_{secondary_event_name}.csv",
            )

            this_output_path = os.path.join(
                self.output_dir, "output", this_year_filename
            )
            bva_last_output_path = os.path.join(
                self.output_dir, "output", last_year_main_filename
            )
            lva_last_output_path = os.path.join(
                self.output_dir, "output", last_year_secondary_filename
            )

            # Save to CSV - NO NORMALIZATION
            self.session_last_filtered_valid_cols_bva.to_csv(
                bva_last_output_path, index=False
            )
            self.session_last_filtered_valid_cols_lva.to_csv(
                lva_last_output_path, index=False
            )
            self.session_this_filtered_valid_cols.to_csv(this_output_path, index=False)

            self.session_file_paths.update(
                {
                    "this_year": this_output_path,
                    "last_year_main": bva_last_output_path,
                    "last_year_secondary": lva_last_output_path,
                }
            )

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
        self.backfill_missing_streams()

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