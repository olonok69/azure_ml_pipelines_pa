import os
import logging
import pandas as pd
from typing import Dict

from pandas.errors import SettingWithCopyWarning
import warnings
warnings.simplefilter(action="ignore", category=(SettingWithCopyWarning))

class ScanProcessor:
    """Process scan data for event analytics - Generic Version."""

    def __init__(self, config: Dict):
        """
        Initialize the processor with configuration.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.output_dir = config.get("output_dir", "output")
        self.mode = config.get("mode", "personal_agendas")
        
        # Get event configuration
        self.event_config = config.get("event", {})
        
        # Get event names
        self.main_event_name = self.event_config.get("main_event_name", "main")
        self.secondary_event_name = self.event_config.get("secondary_event_name", "secondary")

        # Post analysis mode configuration (optional)
        self.post_analysis_config = config.get("post_analysis_mode", {}) if self.mode == "post_analysis" else {}
        self.enhanced_seminars_df_this_year = pd.DataFrame()
        self.seminars_scans_this_year_enhanced = pd.DataFrame()
        
        # Get output file configurations with backward compatibility
        self.output_files = config.get("scan_output_files", {}) or {}

        # Ensure modern blocks exist with sensible defaults so file naming is predictable
        processed_scans_cfg = self.output_files.setdefault("processed_scans", {})
        legacy_scan_cfg = self.output_files.get("scan_data", {}) or {}
        sessions_cfg = self.output_files.setdefault("sessions_visited", {})

        processed_defaults = {
            "last_year_main": legacy_scan_cfg.get("main_event", "scan_bva_past.csv"),
            "last_year_secondary": legacy_scan_cfg.get("secondary_event", "scan_lva_past.csv"),
            "this_year_post": legacy_scan_cfg.get("this_year_post", "scan_this_post.csv"),
        }

        for key, value in processed_defaults.items():
            processed_scans_cfg.setdefault(key, value)

        session_defaults = {
            "main_event": "sessions_visited_last_bva.csv",
            "secondary_event": "sessions_visited_last_lva.csv",
            "this_year_post": "sessions_visited_this_year.csv",
        }

        for key, value in session_defaults.items():
            sessions_cfg.setdefault(key, value)

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "output"), exist_ok=True)

        # Use existing logger instead of configuring a new one
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized ScanProcessor for {self.main_event_name} event with output to {self.output_dir}")

    def load_scan_data(self) -> None:
        """Load all scan data files based on configuration."""
        try:
            # Load session data
            session_this_path = self.config["scan_files"]["session_this"]
            session_past_path_main = self.config["scan_files"]["session_past_main"]
            session_past_path_secondary = self.config["scan_files"]["session_past_secondary"]

            # Load seminar reference data
            seminars_scan_reference_past_path_main = self.config["scan_files"][
                "seminars_scan_reference_past_main"
            ]
            seminars_scans_past_path_main = self.config["scan_files"][
                "seminars_scans_past_main"
            ]
            seminars_scan_reference_past_path_secondary = self.config["scan_files"][
                "seminars_scan_reference_past_secondary"
            ]
            seminars_scans_past_path_secondary = self.config["scan_files"][
                "seminars_scans_past_secondary"
            ]

            # Read session data
            self.session_this = pd.read_csv(session_this_path)
            self.session_past_main = pd.read_csv(session_past_path_main)  # Keep old name for compatibility
            self.session_past_secondary = pd.read_csv(session_past_path_secondary)  # Keep old name for compatibility
            
            # Also create references with old names for backward compatibility
            self.session_past_bva = self.session_past_main
            self.session_past_lva = self.session_past_secondary

            # Read seminar reference data
            self.seminars_scan_reference_past_main = pd.read_csv(
                seminars_scan_reference_past_path_main
            )
            self.seminars_scans_past_main = pd.read_csv(seminars_scans_past_path_main)
            self.seminars_scan_reference_past_secondary = pd.read_csv(
                seminars_scan_reference_past_path_secondary
            )
            self.seminars_scans_past_secondary = pd.read_csv(seminars_scans_past_path_secondary)
            
            # Also create references with old names for backward compatibility
            self.seminars_scan_reference_past_bva = self.seminars_scan_reference_past_main
            self.seminars_scans_past_bva = self.seminars_scans_past_main
            self.seminars_scan_reference_past_lva = self.seminars_scan_reference_past_secondary
            self.seminars_scans_past_lva = self.seminars_scans_past_secondary

            # Load post analysis seminar files (this year) if configured
            if self.mode == "post_analysis":
                pa_scan_files = self.post_analysis_config.get("scan_files", {})
                seminars_scan_reference_this_path = pa_scan_files.get("seminars_scan_reference_this")
                seminars_scans_this_path = pa_scan_files.get("seminars_scans_this")

                if seminars_scan_reference_this_path and seminars_scans_this_path:
                    self.seminars_scan_reference_this_year = pd.read_csv(seminars_scan_reference_this_path)
                    self.seminars_scans_this_year = pd.read_csv(seminars_scans_this_path)
                    self.logger.info(
                        f"Loaded post-analysis seminar scans for this year: reference={len(self.seminars_scan_reference_this_year)} records, scans={len(self.seminars_scans_this_year)} records"
                    )
                else:
                    self.logger.warning("Post-analysis scan files not fully configured; skipping this-year seminar load")
                    self.seminars_scan_reference_this_year = pd.DataFrame()
                    self.seminars_scans_this_year = pd.DataFrame()

            self.logger.info(
                f"Loaded session data: {len(self.session_this)} records this year, "
                f"{len(self.session_past_main)} {self.main_event_name} records last year, "
                f"{len(self.session_past_secondary)} {self.secondary_event_name} records last year"
            )

            self.logger.info(
                f"Loaded seminar scan data: {len(self.seminars_scans_past_main)} {self.main_event_name} records, "
                f"{len(self.seminars_scans_past_secondary)} {self.secondary_event_name} records"
            )

        except Exception as e:
            self.logger.error(f"Error loading scan data: {e}", exc_info=True)
            raise

    def load_registration_data(self) -> None:
        """Load processed registration data from previous step."""
        try:
            # Get output file names from config with fallback to default names
            combined_output_files = self.config.get("output_files", {}).get("combined_demographic_registration", {})
            
            # Load registration data with demographics
            reg_data_last_main_path = os.path.join(
                self.output_dir, "output", 
                combined_output_files.get("last_year_main", "df_reg_demo_last_bva.csv")
            )
            reg_data_last_secondary_path = os.path.join(
                self.output_dir, "output", 
                combined_output_files.get("last_year_secondary", "df_reg_demo_last_lva.csv")
            )

            # Get registration with demographic file names
            reg_with_demo_files = self.config.get("output_files", {}).get("registration_with_demographic", {})
            
            detailed_reg_data_last_main_path = os.path.join(
                self.output_dir,
                "output",
                reg_with_demo_files.get("last_year_main", "Registration_data_with_demographicdata_bva_last.csv"),
            )
            detailed_reg_data_last_secondary_path = os.path.join(
                self.output_dir,
                "output",
                reg_with_demo_files.get("last_year_secondary", "Registration_data_with_demographicdata_lva_last.csv"),
            )

            # Load this-year files for post analysis mode
            if self.mode == "post_analysis":
                this_year_combined_path = os.path.join(
                    self.output_dir,
                    "output",
                    combined_output_files.get("this_year", "df_reg_demo_this.csv")
                )
                this_year_detailed_path = os.path.join(
                    self.output_dir,
                    "output",
                    reg_with_demo_files.get("this_year", "Registration_data_with_demographicdata_bva_this.csv")
                )

                self.reg_data_this_year = pd.read_csv(this_year_combined_path)
                self.df_reg_wdemo_this_year = pd.read_csv(this_year_detailed_path)
            else:
                self.reg_data_this_year = pd.DataFrame()
                self.df_reg_wdemo_this_year = pd.DataFrame()

            # Read registration data
            self.reg_data_last_main = pd.read_csv(reg_data_last_main_path)
            self.reg_data_last_secondary = pd.read_csv(reg_data_last_secondary_path)

            self.df_reg_24_wdemo_data_main = pd.read_csv(detailed_reg_data_last_main_path)
            self.df_reg_24_wdemo_data_secondary = pd.read_csv(detailed_reg_data_last_secondary_path)
            
            # Keep old names for backward compatibility
            self.reg_data_last_bva = self.reg_data_last_main
            self.reg_data_last_lva = self.reg_data_last_secondary
            self.df_reg_24_wdemo_data_bva = self.df_reg_24_wdemo_data_main
            self.df_reg_24_wdemo_data_lva = self.df_reg_24_wdemo_data_secondary

            self.logger.info(
                f"Loaded registration data: {len(self.reg_data_last_main)} {self.main_event_name} records, "
                f"{len(self.reg_data_last_secondary)} {self.secondary_event_name} records"
            )

            self.logger.info(
                f"Loaded detailed registration data: {len(self.df_reg_24_wdemo_data_main)} {self.main_event_name} records, "
                f"{len(self.df_reg_24_wdemo_data_secondary)} {self.secondary_event_name} records"
            )

            if self.mode == "post_analysis":
                self.logger.info(
                    f"Loaded this-year registration data for post analysis: combined={len(self.reg_data_this_year)} records, detailed={len(self.df_reg_wdemo_this_year)} records"
                )

        except Exception as e:
            self.logger.error(f"Error loading registration data: {e}", exc_info=True)
            raise

    @staticmethod
    def clean_text(text):
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

    def enhance_seminar_data(self) -> None:
        """Enhance seminar data with seminar names and create keys for matching."""
        try:
            # Enhance main event seminar data
            self.seminars_scans_past_enhanced_main = pd.merge(
                self.seminars_scans_past_main,
                self.seminars_scan_reference_past_main[["Short Name", "Seminar Name"]],
                on=["Short Name"],
                how="left",
            )

            # Enhance secondary event seminar data
            self.seminars_scans_past_enhanced_secondary = pd.merge(
                self.seminars_scans_past_secondary,
                self.seminars_scan_reference_past_secondary[["Short Name", "Seminar Name"]],
                on=["Short Name"],
                how="left",
            )
            
            # Keep old names for backward compatibility
            self.seminars_scans_past_enhanced_bva = self.seminars_scans_past_enhanced_main
            self.seminars_scans_past_enhanced_lva = self.seminars_scans_past_enhanced_secondary

            # Create keys for matching seminar names
            self.seminars_scans_past_enhanced_main["key_text"] = (
                self.seminars_scans_past_enhanced_main["Seminar Name"].apply(
                    self.clean_text
                )
            )
            self.seminars_scans_past_enhanced_secondary["key_text"] = (
                self.seminars_scans_past_enhanced_secondary["Seminar Name"].apply(
                    self.clean_text
                )
            )

            # Create keys for session titles
            self.session_past_main["key_text"] = self.session_past_main["title"].apply(
                self.clean_text
            )
            self.session_past_secondary["key_text"] = self.session_past_secondary["title"].apply(
                self.clean_text
            )

            if self.mode == "post_analysis" and hasattr(self, "seminars_scans_this_year") and not self.seminars_scans_this_year.empty:
                self.seminars_scans_this_year_enhanced = pd.merge(
                    self.seminars_scans_this_year,
                    self.seminars_scan_reference_this_year[["Short Name", "Seminar Name"]],
                    on=["Short Name"],
                    how="left",
                )
                self.seminars_scans_this_year_enhanced["key_text"] = (
                    self.seminars_scans_this_year_enhanced["Seminar Name"].apply(self.clean_text)
                )
                # Align session key text
                if "key_text" not in self.session_this.columns:
                    self.session_this["key_text"] = self.session_this["title"].apply(self.clean_text)
                self.logger.info(
                    f"Enhanced post-analysis seminar data with {len(self.seminars_scans_this_year_enhanced)} this-year records"
                )
            else:
                self.seminars_scans_this_year_enhanced = pd.DataFrame()

            self.logger.info(
                f"Enhanced seminar data with seminar names: {len(self.seminars_scans_past_enhanced_main)} {self.main_event_name} records, "
                f"{len(self.seminars_scans_past_enhanced_secondary)} {self.secondary_event_name} records"
            )

        except Exception as e:
            self.logger.error(f"Error enhancing seminar data: {e}", exc_info=True)
            raise

    def analyze_data_intersections(self) -> None:
        """Analyze intersections between different datasets."""
        try:
            # Main event seminar and session intersections
            main_seminar_keys = set(
                list(self.seminars_scans_past_enhanced_main["key_text"].unique())
            )
            main_session_keys = set(list(self.session_past_main["key_text"].unique()))
            main_keys_intersection = main_seminar_keys.intersection(main_session_keys)

            # Secondary event seminar and session intersections
            secondary_seminar_keys = set(
                list(self.seminars_scans_past_enhanced_secondary["key_text"].unique())
            )
            secondary_session_keys = set(list(self.session_past_secondary["key_text"].unique()))
            secondary_keys_intersection = secondary_seminar_keys.intersection(secondary_session_keys)

            # Main event registration and seminar badge ID intersections
            main_reg_badges = set(list(self.reg_data_last_main["BadgeId"].unique()))
            main_seminar_badges = set(
                list(self.seminars_scans_past_enhanced_main["Badge Id"].unique())
            )
            main_badges_intersection = main_reg_badges.intersection(main_seminar_badges)

            # Secondary event registration and seminar badge ID intersections
            secondary_reg_badges = set(list(self.reg_data_last_secondary["BadgeId"].unique()))
            secondary_seminar_badges = set(
                list(self.seminars_scans_past_enhanced_secondary["Badge Id"].unique())
            )
            secondary_badges_intersection = secondary_reg_badges.intersection(secondary_seminar_badges)

            # Log statistics
            self.logger.info(
                f"{self.main_event_name} seminar-session intersection: {len(main_keys_intersection)} out of {len(main_seminar_keys)} seminar keys and {len(main_session_keys)} session keys"
            )
            self.logger.info(
                f"{self.secondary_event_name} seminar-session intersection: {len(secondary_keys_intersection)} out of {len(secondary_seminar_keys)} seminar keys and {len(secondary_session_keys)} session keys"
            )
            self.logger.info(
                f"{self.main_event_name} registration-seminar badge intersection: {len(main_badges_intersection)} out of {len(main_reg_badges)} registration badges and {len(main_seminar_badges)} seminar badges"
            )
            self.logger.info(
                f"{self.secondary_event_name} registration-seminar badge intersection: {len(secondary_badges_intersection)} out of {len(secondary_reg_badges)} registration badges and {len(secondary_seminar_badges)} seminar badges"
            )

            if self.mode == "post_analysis" and not self.seminars_scans_this_year_enhanced.empty:
                this_seminar_keys = set(self.seminars_scans_this_year_enhanced["key_text"].unique())
                this_session_keys = set(self.session_this["key_text"].unique())
                current_keys_intersection = this_seminar_keys.intersection(this_session_keys)
                self.logger.info(
                    f"This-year seminar-session intersection (post analysis): {len(current_keys_intersection)} out of {len(this_seminar_keys)} seminar keys and {len(this_session_keys)} session keys"
                )

                if not self.reg_data_this_year.empty:
                    this_reg_badges = set(self.reg_data_this_year["BadgeId"].unique())
                    this_seminar_badges = set(self.seminars_scans_this_year_enhanced["Badge Id"].unique())
                    this_badges_intersection = this_reg_badges.intersection(this_seminar_badges)
                    self.logger.info(
                        f"This-year registration-seminar badge intersection: {len(this_badges_intersection)} out of {len(this_reg_badges)} registration badges and {len(this_seminar_badges)} seminar badges"
                    )

        except Exception as e:
            self.logger.error(f"Error analyzing data intersections: {e}", exc_info=True)
            raise

    def add_demographics_to_seminars(
        self, registration_df: pd.DataFrame, seminars_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add demographic data to seminar scans dataframe for matching badge IDs.

        Args:
            registration_df: DataFrame containing registration and demographic data
            seminars_df: DataFrame containing seminar attendance data

        Returns:
            DataFrame with demographic data added to seminar scans
        """
        try:
            # Create copies to avoid modifying original dataframes
            registration_df = registration_df.copy()
            seminars_df = seminars_df.copy()

            # Rename 'BadgeId' column in registration dataframe to match 'Badge Id' in seminars dataframe
            registration_df = registration_df.rename(columns={"BadgeId": "Badge Id"})

            # Select demographic columns to include
            demographic_columns = [
                "Badge Id",
                "Title",
                "Forename",
                "Surname",
                "Email",
                "Company",
                "JobTitle",
                "Country",
                "BadgeType",
                "Email_domain",
            ]

            # Check if columns exist and filter to available columns
            available_cols = [
                col for col in demographic_columns if col in registration_df.columns
            ]
            reg_slim = registration_df[available_cols]

            # Merge the seminars dataframe with the registration dataframe
            enhanced_seminars = pd.merge(
                seminars_df, reg_slim, on="Badge Id", how="left"
            )

            # Log statistics
            total_seminars = len(seminars_df)
            matched_seminars = enhanced_seminars["Email"].notna().sum()
            match_percentage = (
                (matched_seminars / total_seminars * 100) if total_seminars > 0 else 0
            )

            self.logger.info(f"Total seminar records: {total_seminars}")
            self.logger.info(
                f"Records with matching badge IDs: {matched_seminars} ({match_percentage:.2f}%)"
            )

            return enhanced_seminars

        except Exception as e:
            self.logger.error(
                f"Error adding demographics to seminars: {e}", exc_info=True
            )
            raise

    def enhance_seminar_data_with_demographics(self) -> None:
        """Enhance seminar data with demographic information."""
        try:
            # Add demographics to main event seminar data
            self.enhanced_seminars_df_main = self.add_demographics_to_seminars(
                self.df_reg_24_wdemo_data_main, self.seminars_scans_past_enhanced_main
            )

            # Add demographics to secondary event seminar data
            self.enhanced_seminars_df_secondary = self.add_demographics_to_seminars(
                self.df_reg_24_wdemo_data_secondary, self.seminars_scans_past_enhanced_secondary
            )
            
            # Keep old names for backward compatibility
            self.enhanced_seminars_df_bva = self.enhanced_seminars_df_main
            self.enhanced_seminars_df_lva = self.enhanced_seminars_df_secondary

            self.logger.info("Successfully enhanced seminar data with demographics")

        except Exception as e:
            self.logger.error(
                f"Error enhancing seminar data with demographics: {e}", exc_info=True
            )
            raise

    def save_processed_data(self) -> None:
        """Save all processed data to CSV files."""
        try:
            processed_scans_config = self.output_files.get("processed_scans", {})
            scan_data_config = self.output_files.get("scan_data", {})
            sessions_visited_config = self.output_files.get("sessions_visited", {})

            def _resolve_scan_filename(primary_key: str, legacy_key: str, default_name: str) -> str:
                if processed_scans_config.get(primary_key):
                    return processed_scans_config[primary_key]
                if scan_data_config.get(legacy_key):
                    return scan_data_config[legacy_key]
                return default_name

            main_scan_filename = _resolve_scan_filename("last_year_main", "main_event", "scan_bva_past.csv")
            secondary_scan_filename = _resolve_scan_filename("last_year_secondary", "secondary_event", "scan_lva_past.csv")
            this_year_post_filename = _resolve_scan_filename("this_year_post", "this_year_post", "scan_this_post.csv")
            this_year_filename = processed_scans_config.get("this_year")

            main_scan_output_path = os.path.join(self.output_dir, "output", main_scan_filename)
            secondary_scan_output_path = os.path.join(self.output_dir, "output", secondary_scan_filename)

            # Save enhanced seminar data with demographics
            main_sessions_output_path = os.path.join(
                self.output_dir, "output",
                sessions_visited_config.get("main_event", "sessions_visited_last_bva.csv")
            )
            secondary_sessions_output_path = os.path.join(
                self.output_dir, "output",
                sessions_visited_config.get("secondary_event", "sessions_visited_last_lva.csv")
            )

            self.seminars_scans_past_enhanced_main.to_csv(main_scan_output_path, index=False)
            self.seminars_scans_past_enhanced_secondary.to_csv(secondary_scan_output_path, index=False)
            self.enhanced_seminars_df_main.to_csv(main_sessions_output_path, index=False)
            self.enhanced_seminars_df_secondary.to_csv(secondary_sessions_output_path, index=False)

            if this_year_filename and not self.seminars_scans_this_year_enhanced.empty:
                this_year_scan_path = os.path.join(self.output_dir, "output", this_year_filename)
                self.seminars_scans_this_year_enhanced.to_csv(this_year_scan_path, index=False)
                self.logger.info(
                    "Saved current-year processed scans to %s", this_year_filename
                )
            elif this_year_filename:
                self.logger.warning(
                    "Configured processed_scans.this_year='%s' but no current-year scan data was loaded",
                    this_year_filename,
                )

            if self.mode == "post_analysis" and not self.seminars_scans_this_year_enhanced.empty:
                this_year_scan_output_path = os.path.join(
                    self.output_dir,
                    "output",
                    this_year_post_filename,
                )
                this_year_sessions_output_path = os.path.join(
                    self.output_dir,
                    "output",
                    sessions_visited_config.get("this_year_post", "sessions_visited_this_year.csv"),
                )

                this_year_enhanced = self.add_demographics_to_seminars(
                    self.df_reg_wdemo_this_year,
                    self.seminars_scans_this_year_enhanced,
                ) if not self.df_reg_wdemo_this_year.empty else self.seminars_scans_this_year_enhanced.copy()

                # Ensure key_text column exists for downstream matching
                if "key_text" not in this_year_enhanced.columns and "Seminar Name" in this_year_enhanced.columns:
                    this_year_enhanced["key_text"] = this_year_enhanced["Seminar Name"].apply(self.clean_text)

                self.seminars_scans_this_year_enhanced.to_csv(this_year_scan_output_path, index=False)
                this_year_enhanced.to_csv(this_year_sessions_output_path, index=False)
                self.enhanced_seminars_df_this_year = this_year_enhanced

                # Consistency check with personal agenda outputs
                expected_columns = set(self.enhanced_seminars_df_main.columns)
                current_columns = set(this_year_enhanced.columns)
                if expected_columns != current_columns:
                    self.logger.warning(
                        "Post-analysis sessions output columns differ from personal_agendas baseline. Missing: %s | Additional: %s",
                        expected_columns - current_columns,
                        current_columns - expected_columns,
                    )
                else:
                    self.logger.info("Post-analysis sessions output matches personal_agendas column schema")

                self.logger.info(
                    f"Saved post-analysis scan data: {len(self.seminars_scans_this_year_enhanced)} this-year records, with demographics records={len(this_year_enhanced)}"
                )

            self.logger.info(f"Saved processed scan data to {self.output_dir}/output/")
            self.logger.info(
                f"{self.main_event_name} scan data: {len(self.seminars_scans_past_enhanced_main)} records"
            )
            self.logger.info(
                f"{self.secondary_event_name} scan data: {len(self.seminars_scans_past_enhanced_secondary)} records"
            )
            self.logger.info(
                f"{self.main_event_name} sessions with demographics: {len(self.enhanced_seminars_df_main)} records"
            )
            self.logger.info(
                f"{self.secondary_event_name} sessions with demographics: {len(self.enhanced_seminars_df_secondary)} records"
            )

            if self.mode == "post_analysis" and not self.seminars_scans_this_year_enhanced.empty:
                self.logger.info(
                    f"Post-analysis sessions with demographics (this year): {len(this_year_enhanced)} records"
                )

        except Exception as e:
            self.logger.error(f"Error saving processed data: {e}", exc_info=True)
            raise

    def process(self) -> None:
        """Execute the full scan data processing workflow."""
        self.logger.info("Starting scan data processing workflow")

        self.load_scan_data()
        self.load_registration_data()
        self.enhance_seminar_data()
        self.analyze_data_intersections()
        self.enhance_seminar_data_with_demographics()
        self.save_processed_data()

        self.logger.info("Scan data processing workflow completed successfully")