import os
import json
import string
import logging
import pandas as pd
import functools
from typing import Dict, List, Set
from datetime import datetime
from pandas import json_normalize
from fuzzywuzzy import process
import re
from difflib import SequenceMatcher
from pandas.errors import SettingWithCopyWarning
import warnings
warnings.simplefilter(action="ignore", category=(SettingWithCopyWarning))
# Configure logger
logger = logging.getLogger(__name__)


class DataLoader:
    """Handle loading data from different sources."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_json(self, file_path: str) -> List[Dict]:
        """
        Load data from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            List of dictionaries containing the data
        """
        self.logger.info(f"Loading JSON data from {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logger.info(
                f"Successfully loaded {len(data)} records from {file_path}"
            )
            return data
        except Exception as e:
            self.logger.error(
                f"Error loading JSON data from {file_path}: {e}", exc_info=True
            )
            raise


# Optional decorator for logging function entry/exit
def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        logger.debug(f"Exiting {func.__name__}")
        return result

    return wrapper


class RegistrationProcessor:
    """Process registration data for event analytics."""

    def __init__(self, config: Dict):
        """
        Initialize the processor with configuration.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.output_dir = config.get("output_dir", "output")
        
        # Get event configuration
        self.event_config = config.get("event", {})
        
        # Get event names and shows configuration
        self.main_event_name = self.event_config.get("main_event_name", "main")
        self.secondary_event_name = self.event_config.get("secondary_event_name", "secondary")
        
        # Handle shows_this_year - flatten if it's a list of lists (for ecomm)
        shows_this_year_raw = self.event_config.get("shows_this_year", [])
        self.shows_this_year = self._flatten_show_list(shows_this_year_raw)
        
        # Handle shows_this_year_exclude - flatten if it's a list of lists (for ecomm)
        shows_this_year_exclude_raw = self.event_config.get("shows_this_year_exclude", [])
        self.shows_this_year_exclude = self._flatten_show_list(shows_this_year_exclude_raw)
        
        self.shows_last_year_main = self.event_config.get("shows_last_year_main", [])
        self.shows_last_year_secondary = self.event_config.get("shows_last_year_secondary", [])

        # Get output file configurations with backward compatibility
        self.output_files = config.get("output_files", {})
        
        # Backward compatibility: if output_files section doesn't exist, use default names
        if not self.output_files:
            self.output_files = {
                "professions_list": "list_of_professions.csv",
                "specializations": "specializations.json",
                "job_roles": "job_roles.json",
                "raw_data": {
                    "main_event_registration": "Registration_data_bva.csv",
                    "secondary_event_registration": "Registration_data_lva.csv",
                    "main_event_demographic": "Registration_demographicdata_bva.csv",
                    "secondary_event_demographic": "Registration_demographicdata_lva.csv"
                },
                "processed_data": {
                    "this_year": "Registration_data_bva_25_only_valid.csv",
                    "last_year_main": "Registration_data_bva_24_only_valid.csv",
                    "returning_main": "Registration_data_bva_24_25_only_valid.csv",
                    "returning_secondary": "Registration_data_lva_24_25_only_valid.csv"
                }
            }

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "csv"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "output"), exist_ok=True)

        # Use existing logger instead of configuring a new one
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initialized RegistrationProcessor for {self.main_event_name} event with output to {self.output_dir}"
        )

    def _flatten_show_list(self, show_list: List) -> List[str]:
        """
        Flatten a list that might contain nested lists or strings.
        
        Args:
            show_list: List that might contain strings or nested lists
            
        Returns:
            Flattened list of strings
        """
        if not show_list:
            return []
            
        flattened = []
        for item in show_list:
            if isinstance(item, list):
                # If item is a list, extend with its contents
                flattened.extend(item)
            else:
                # If item is a string, append it
                flattened.append(item)
        
        return flattened

    def load_data(self) -> None:
        """Load all registration data files based on configuration."""
        loader = DataLoader()

        # Load main event registration data
        main_event_reg_file = self.config["input_files"]["main_event_registration"]
        self.main_event_reg_data = loader.load_json(main_event_reg_file)

        # Load secondary event registration data
        secondary_event_reg_file = self.config["input_files"]["secondary_event_registration"]
        self.secondary_event_reg_data = loader.load_json(secondary_event_reg_file)

        # Load main event demographic data
        main_event_demo_file = self.config["input_files"]["main_event_demographic"]
        self.main_event_demo_data = loader.load_json(main_event_demo_file)

        # Load secondary event demographic data
        secondary_event_demo_file = self.config["input_files"]["secondary_event_demographic"]
        self.secondary_event_demo_data = loader.load_json(secondary_event_demo_file)

        # Convert to dataframes
        self.df_bva = pd.json_normalize(self.main_event_reg_data)  # Keep old naming for compatibility
        self.df_lvs = pd.json_normalize(self.secondary_event_reg_data)  # Keep old naming for compatibility
        self.df_bva_demo = pd.json_normalize(self.main_event_demo_data)  # Keep old naming for compatibility
        self.df_lvs_demo = pd.json_normalize(self.secondary_event_demo_data)  # Keep old naming for compatibility

        self.logger.info(
            f"Loaded {len(self.df_bva)} {self.main_event_name} registration records and {len(self.df_lvs)} {self.secondary_event_name} registration records"
        )
        self.logger.info(
            f"Loaded {len(self.df_bva_demo)} {self.main_event_name} demographic records and {len(self.df_lvs_demo)} {self.secondary_event_name} demographic records"
        )

    def save_initial_data(self) -> None:
        """Save the initial raw data to CSV files."""
        # Get output file names from config with fallback to default names
        raw_data_config = self.output_files.get("raw_data", {})
        main_event_csv = os.path.join(self.output_dir, "csv", 
                                     raw_data_config.get("main_event_registration", "Registration_data_bva.csv"))
        secondary_event_csv = os.path.join(self.output_dir, "csv", 
                                          raw_data_config.get("secondary_event_registration", "Registration_data_lva.csv"))
        main_event_demo_csv = os.path.join(self.output_dir, "csv", 
                                          raw_data_config.get("main_event_demographic", "Registration_demographicdata_bva.csv"))
        secondary_event_demo_csv = os.path.join(self.output_dir, "csv", 
                                               raw_data_config.get("secondary_event_demographic", "Registration_demographicdata_lva.csv"))

        self.df_bva.to_csv(main_event_csv, index=False)
        self.df_lvs.to_csv(secondary_event_csv, index=False)
        self.df_bva_demo.to_csv(main_event_demo_csv, index=False)
        self.df_lvs_demo.to_csv(secondary_event_demo_csv, index=False)

        self.logger.info(f"Saved initial data to CSV files in {self.output_dir}/csv/")


    def apply_question_text_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply question text corrections based on configuration mapping.
        
        Args:
            df: DataFrame containing QuestionText column to correct
            
        Returns:
            DataFrame with corrected QuestionText values
        """
        # Check if question_text_corrections section exists in config
        corrections = self.config.get("question_text_corrections", {})
        
        if not corrections:
            # No corrections configured, return dataframe unchanged
            return df
        
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Apply each correction
        corrections_applied = 0
        for incorrect_text, correct_text in corrections.items():
            # Count how many rows will be affected
            affected_rows = df['QuestionText'] == incorrect_text
            num_affected = affected_rows.sum()
            
            if num_affected > 0:
                df.loc[affected_rows, 'QuestionText'] = correct_text
                corrections_applied += 1
                self.logger.info(
                    f"Corrected question text: '{incorrect_text}' -> '{correct_text}' ({num_affected} rows affected)"
                )
        
        if corrections_applied > 0:
            self.logger.info(f"Applied {corrections_applied} question text corrections")
        else:
            self.logger.info("No question text corrections were needed")
        
        return df

    def preprocess_data(self) -> None:
        """Preprocess the registration data."""
        # Clean email and name fields - create copies to avoid warnings
        self.df_bva = self.df_bva.dropna(subset=["Email", "Forename", "Surname"]).copy()
        self.df_lvs = self.df_lvs.dropna(subset=["Email", "Forename", "Surname"]).copy()

        # Normalize job titles - use loc to avoid SettingWithCopyWarning
        self.df_bva.loc[:, "JobTitle"] = self.df_bva["JobTitle"].str.lower()
        self.df_bva.loc[:, "JobTitle"] = self.df_bva["JobTitle"].fillna("NA")

        # Save unique job titles for reference
        job_titles = pd.DataFrame(
            data=list(self.df_bva["JobTitle"].unique()), columns=["JobTitle"]
        )
        job_titles_file = self.output_files.get("professions_list", "list_of_professions.csv")
        job_titles.to_csv(
            os.path.join(self.output_dir, job_titles_file), index=False
        )

        # Preprocess demographic data
        # Remove punctuation from QuestionText
        self.df_bva_demo.loc[:, "QuestionText"] = self.df_bva_demo[
            "QuestionText"
        ].apply(self.remove_punctuation)
        self.df_lvs_demo.loc[:, "QuestionText"] = self.df_lvs_demo[
            "QuestionText"
        ].apply(self.remove_punctuation)

        # Apply question text corrections after removing punctuation
        self.df_bva_demo = self.apply_question_text_corrections(self.df_bva_demo)
        self.df_lvs_demo = self.apply_question_text_corrections(self.df_lvs_demo)

        # Create backup of AnswerText
        self.df_bva_demo.loc[:, "AnswerText_backup"] = self.df_bva_demo[
            "AnswerText"
        ].astype(str)
        self.df_lvs_demo.loc[:, "AnswerText_backup"] = self.df_lvs_demo[
            "AnswerText"
        ].astype(str)

        self.logger.info(
            f"Preprocessed data - {self.main_event_name}: {len(self.df_bva)} records, {self.secondary_event_name}: {len(self.df_lvs)} records"
        )
        self.logger.info(f"Found {len(job_titles)} unique job titles")

    @staticmethod
    def remove_punctuation(text):
        """
        Removes punctuation from a string, excluding hyphens and forward slashes.

        Args:
            text: Input string

        Returns:
            String with punctuation removed
        """
        if not isinstance(text, str):
            return text

        custom_punctuation = (
            string.punctuation.replace("-", "").replace("/", "") + "''" "…" + "â€™Â"
        )
        return text.translate(str.maketrans("", "", custom_punctuation))

    @staticmethod
    def calculate_date_difference(
        df: pd.DataFrame, date_column: str, given_date_str: str
    ) -> pd.DataFrame:
        """
        Calculate the number of days between registration date and event date.

        Args:
            df: DataFrame containing registration data
            date_column: Name of the column containing registration dates
            given_date_str: Event date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with an additional 'Days_since_registration' column
        """
        logger = logging.getLogger(__name__)

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_copy = df.copy()

        # Check if date_column exists in the DataFrame
        if date_column not in df_copy.columns:
            logger.warning(
                f"Date column '{date_column}' not found in DataFrame. Adding dummy values."
            )
            df_copy.loc[:, "Days_since_registration"] = 0
            return df_copy

        # Convert the given date to datetime
        given_date = datetime.strptime(given_date_str, "%Y-%m-%d")

        # Convert the column to datetime, with error handling
        try:
            # Handle potential NaN values
            mask = df_copy[date_column].notna()
            if not mask.any():
                logger.warning(
                    f"No valid dates found in '{date_column}'. Adding dummy values."
                )
                df_copy.loc[:, "Days_since_registration"] = 0
                return df_copy

            # Initialize Days_since_registration column with default value
            df_copy.loc[:, "Days_since_registration"] = 0

            # For each row with a non-null date, calculate days difference
            for idx in df_copy[mask].index:
                try:
                    date_val = df_copy.loc[idx, date_column]
                    # Make sure we have a proper datetime object
                    if not isinstance(date_val, datetime):
                        date_val = pd.to_datetime(date_val, errors="coerce")

                    if pd.notna(date_val):
                        # Calculate days difference directly without using .dt accessor
                        days_diff = (given_date - date_val).days
                        df_copy.loc[idx, "Days_since_registration"] = days_diff
                except Exception as e:
                    logger.warning(f"Error processing date for row {idx}: {e}")

        except Exception as e:
            logger.error(f"Error calculating date difference: {e}", exc_info=True)
            # Keep default values to continue processing

        return df_copy

    @staticmethod
    def create_unique_id(row: pd.Series) -> str:
        """
        Create a unique identifier using name and email.

        Args:
            row: DataFrame row containing Forename, Surname, and Email

        Returns:
            String concatenation of lowercase forename, surname and email
        """
        return (
            f"{row['Forename'].lower()}_{row['Surname'].lower()}_{row['Email'].lower()}"
        )

    @staticmethod
    def extract_email_domain(email: str) -> str:
        """
        Extract the domain part from an email address.

        Args:
            email: Email address

        Returns:
            Domain part of the email
        """
        return email.split("@")[-1]

    def flag_returning_visitors(
        self, df: pd.DataFrame, returning_visitors: Set[str]
    ) -> pd.DataFrame:
        """
        Flag visitors who attended the previous year's event.

        Args:
            df: DataFrame containing registration data
            returning_visitors: Set of unique IDs of returning visitors

        Returns:
            DataFrame with an additional 'assist_year_before' column
        """
        # Create a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()

        # Use loc to set values
        df_copy.loc[:, "assist_year_before"] = df_copy.apply(
            lambda row: 1 if row["id_both_years"] in returning_visitors else 0, axis=1
        )
        return df_copy

    def split_and_process_bva_data(self) -> None:
        """Split main event data by year and process it."""
        # Define valid badge types
        valid_badge_types = self.config.get(
            "valid_badge_types", ["Delegate", "Delegate - Group"]
        )

        # Split registration data by year
        # Use the flattened shows_this_year list
        self.df_bva_this_year = self.df_bva[self.df_bva.ShowRef.isin(self.shows_this_year)]
        
        # For last year, exclude both this year's shows AND the exclude list
        all_shows_to_exclude = self.shows_this_year + self.shows_this_year_exclude
        self.df_bva_last_year = self.df_bva[~self.df_bva.ShowRef.isin(all_shows_to_exclude)]

        # Split demographic data by year
        self.df_bva_demo_this_year = self.df_bva_demo[
            self.df_bva_demo.showref.isin(self.shows_this_year)
        ]
        self.df_bva_demo_last_year = self.df_bva_demo[
            ~self.df_bva_demo.showref.isin(all_shows_to_exclude)
        ]

        # Process secondary event data
        # For ecomm: df_lvs_last_year should contain shows from shows_this_year_exclude (like TFM24)
        # For vet: df_lvs_last_year should contain shows from shows_last_year_secondary (like LVS2024)
        if self.shows_this_year_exclude:
            # For ecomm: get records from the exclude list (e.g., TFM24)
            self.df_lvs_last_year = self.df_lvs[
                self.df_lvs.ShowRef.isin(self.shows_this_year_exclude)
            ]
            # Filter by valid badge types
            self.df_lvs_last_year = self.df_lvs_last_year[
                self.df_lvs_last_year["BadgeType"].isin(valid_badge_types)
            ]
            
            # Also update the demographic data for secondary event
            self.df_lvs_demo = self.df_lvs_demo[
                self.df_lvs_demo.showref.isin(self.shows_this_year_exclude)
            ]
        else:
            # For vet: use the existing logic with valid badge types only
            self.df_lvs_last_year = self.df_lvs[
                self.df_lvs["BadgeType"].isin(valid_badge_types)
            ]

        self.logger.info(
            f"Split {self.main_event_name} data: {len(self.df_bva_this_year)} records for this year, {len(self.df_bva_last_year)} records for last year"
        )
        self.logger.info(
            f"Split {self.main_event_name} demo data: {len(self.df_bva_demo_this_year)} records for this year, {len(self.df_bva_demo_last_year)} records for last year"
        )

        # Filter by valid badge types
        self.df_bva_this_year = self.df_bva_this_year[
            self.df_bva_this_year["BadgeType"].isin(valid_badge_types)
        ]
        self.df_bva_last_year = self.df_bva_last_year[
            self.df_bva_last_year["BadgeType"].isin(valid_badge_types)
        ]

        # Remove duplicates
        self.df_bva_this_year = self.df_bva_this_year.drop_duplicates(
            subset="BadgeId", keep="first"
        )
        self.df_bva_last_year = self.df_bva_last_year.drop_duplicates(
            subset="BadgeId", keep="first"
        )
        self.df_lvs_last_year = self.df_lvs_last_year.drop_duplicates(
            subset="BadgeId", keep="first"
        )

        self.logger.info(
            f"After filtering and deduplication: {self.main_event_name} this year: {len(self.df_bva_this_year)}, {self.main_event_name} last year: {len(self.df_bva_last_year)}, {self.secondary_event_name} last year: {len(self.df_lvs_last_year)}"
        )

        # Create ID for matching visitors across years
        # Use loc to avoid SettingWithCopyWarning
        for df_name in ["df_bva_this_year", "df_bva_last_year", "df_lvs_last_year"]:
            df = getattr(self, df_name)
            df_copy = df.copy()
            df_copy.loc[:, "id_both_years"] = df.apply(self.create_unique_id, axis=1)
            df_copy.loc[:, "Email_domain"] = df["Email"].apply(
                self.extract_email_domain
            )
            setattr(self, df_name, df_copy)

    def identify_returning_visitors(self) -> None:
        """Identify visitors who attended previous year's events."""
        # Get unique IDs for each year
        ids_bva_last_year = set(self.df_bva_last_year["id_both_years"].unique())
        ids_bva_this_year = set(self.df_bva_this_year["id_both_years"].unique())
        ids_lvs_last_year = set(self.df_lvs_last_year["id_both_years"].unique())

        # Find intersections - keep old variable names for compatibility
        self.bva_returning_visitors = ids_bva_this_year.intersection(ids_bva_last_year)
        self.lvs_returning_visitors = ids_bva_this_year.intersection(ids_lvs_last_year)

        # Union of all returning visitors
        self.all_returning_visitors = self.bva_returning_visitors.union(
            self.lvs_returning_visitors
        )

        self.logger.info(
            f"Identified {len(self.bva_returning_visitors)} returning visitors from {self.main_event_name} and {len(self.lvs_returning_visitors)} from {self.secondary_event_name}"
        )
        self.logger.info(
            f"Total unique returning visitors: {len(self.all_returning_visitors)}"
        )

        # Create dataframes for returning visitors - keep old variable names
        self.df_bva_returning = self.df_bva_last_year[
            self.df_bva_last_year["id_both_years"].isin(self.bva_returning_visitors)
        ]
        self.df_lvs_returning = self.df_lvs_last_year[
            self.df_lvs_last_year["id_both_years"].isin(self.lvs_returning_visitors)
        ]

    def add_badge_history(self) -> None:
        """Add badge history information to current year data."""
        # Create index dataframes
        badge_last_year_bva = self.df_bva_returning[["BadgeId", "id_both_years"]].copy()
        badge_last_year_lvs = self.df_lvs_returning[["BadgeId", "id_both_years"]].copy()

        # Get column names from config
        badge_history_cols = self.config.get("badge_history_columns", {})
        main_col = badge_history_cols.get("main_event", "BadgeId_last_year_bva")
        secondary_col = badge_history_cols.get("secondary_event", "BadgeId_last_year_lva")

        # Rename columns
        badge_last_year_bva.columns = [main_col, "id_both_years"]
        badge_last_year_lvs.columns = [secondary_col, "id_both_years"]

        # Merge with current year data
        self.df_bva_this_year = pd.merge(
            self.df_bva_this_year, badge_last_year_bva, on=["id_both_years"], how="left"
        )
        self.df_bva_this_year = pd.merge(
            self.df_bva_this_year, badge_last_year_lvs, on=["id_both_years"], how="left"
        )

        # Fill missing values - use loc to avoid SettingWithCopyWarning
        self.df_bva_this_year.loc[:, main_col] = self.df_bva_this_year[main_col].fillna("NA")
        self.df_bva_this_year.loc[:, secondary_col] = self.df_bva_this_year[secondary_col].fillna("NA")

        self.logger.info("Added badge history information to current year data")
        # Log the number of matched records for verification
        self.logger.info(
            f"Number of unique {main_col} values: {len(self.df_bva_this_year[main_col].unique())}"
        )
        self.logger.info(
            f"Number of unique {secondary_col} values: {len(self.df_bva_this_year[secondary_col].unique())}"
        )

    def calculate_event_dates(self) -> None:
        """Calculate days until event for all dataframes."""
        event_date_this_year = self.config.get("event_date_this_year", "2025-06-12")
        event_date_last_year = self.config.get("event_date_last_year", "2024-06-12")

        self.df_bva_this_year = self.calculate_date_difference(
            self.df_bva_this_year, "RegistrationDate", event_date_this_year
        )
        self.df_bva_last_year = self.calculate_date_difference(
            self.df_bva_last_year, "RegistrationDate", event_date_this_year
        )
        self.df_bva_returning = self.calculate_date_difference(
            self.df_bva_returning, "RegistrationDate", event_date_last_year
        )
        self.df_lvs_returning = self.calculate_date_difference(
            self.df_lvs_returning, "RegistrationDate", event_date_last_year
        )

        self.logger.info("Calculated days until event for all datasets")

    def flag_all_returning_visitors(self) -> None:
        """Flag all returning visitors in current year data."""
        self.df_bva_this_year = self.flag_returning_visitors(
            self.df_bva_this_year, self.all_returning_visitors
        )
        self.logger.info(
            f"Returning visitors: {self.df_bva_this_year['assist_year_before'].sum()} out of {len(self.df_bva_this_year)}"
        )

    def select_and_save_final_data(self) -> None:
        """Select relevant columns and save final processed data."""
        # Get badge history column names from config
        badge_history_cols = self.config.get("badge_history_columns", {})
        main_col = badge_history_cols.get("main_event", "BadgeId_last_year_bva")
        secondary_col = badge_history_cols.get("secondary_event", "BadgeId_last_year_lva")

        # Define columns to keep
        columns_this_year = [
            "Email",
            "Email_domain",
            "Company",
            "JobTitle",
            "Country",
            "BadgeType",
            "ShowRef",
            "BadgeId",
            "Source",
            "Days_since_registration",
            "assist_year_before",
            main_col,
            secondary_col,
        ]

        columns_last_year = [
            "Email",
            "Email_domain",
            "Company",
            "JobTitle",
            "Country",
            "BadgeType",
            "ShowRef",
            "BadgeId",
            "Source",
            "Days_since_registration",
        ]

        # Select columns
        df_this_year_final = self.df_bva_this_year[columns_this_year]
        df_last_year_final = self.df_bva_last_year[columns_last_year]
        df_bva_returning_final = self.df_bva_returning[columns_last_year]
        df_lvs_returning_final = self.df_lvs_returning[columns_last_year]

        # Get output file names from config with fallback to default names
        processed_files = self.output_files.get("processed_data", {})
        output_files = {
            "this_year": os.path.join(
                self.output_dir, "csv", processed_files.get("this_year", "Registration_data_bva_25_only_valid.csv")
            ),
            "last_year": os.path.join(
                self.output_dir, "csv", processed_files.get("last_year_main", "Registration_data_bva_24_only_valid.csv")
            ),
            "main_returning": os.path.join(
                self.output_dir, "csv", processed_files.get("returning_main", "Registration_data_bva_24_25_only_valid.csv")
            ),
            "secondary_returning": os.path.join(
                self.output_dir, "csv", processed_files.get("returning_secondary", "Registration_data_lva_24_25_only_valid.csv")
            ),
        }

        df_this_year_final.to_csv(output_files["this_year"], index=False)
        df_last_year_final.to_csv(output_files["last_year"], index=False)
        df_bva_returning_final.to_csv(output_files["main_returning"], index=False)
        df_lvs_returning_final.to_csv(output_files["secondary_returning"], index=False)

        self.logger.info(f"Saved final processed data to {self.output_dir}/csv/")

    def save_demographic_data(self) -> None:
        """Save demographic data to CSV files."""
        # Get output file names from config
        demo_files = self.output_files.get("demographic_data", {})
        
        demo_this_file = os.path.join(
            self.output_dir, "csv", demo_files.get("this_year", "Registration_demographicdata_this_year_raw.csv")
        )
        demo_last_main_file = os.path.join(
            self.output_dir, "csv", demo_files.get("last_year_main", "Registration_demographicdata_last_year_main_raw.csv")
        )
        demo_last_secondary_file = os.path.join(
            self.output_dir, "csv", demo_files.get("last_year_secondary", "Registration_demographicdata_last_year_secondary_raw.csv")
        )

        self.df_bva_demo_this_year.to_csv(demo_this_file, index=False)
        self.df_bva_demo_last_year.to_csv(demo_last_main_file, index=False)
        self.df_lvs_demo.to_csv(demo_last_secondary_file, index=False)

        self.logger.info("Saved demographic data to CSV files")

    def extract_specialization_data(self) -> None:
        """Extract and analyze specialization data."""
        # Get the questions to keep from config
        questions_to_keep = self.config.get("questions_to_keep", {})
        list_keep_this = questions_to_keep.get("current", [])
        list_keep_past = questions_to_keep.get("past", [])

        # Extract specialization data
        df_spe_this = self.df_bva_demo_this_year[
            self.df_bva_demo_this_year["QuestionText"] == list_keep_this[0]
        ]
        df_spe_last_bva = self.df_bva_demo_last_year[
            self.df_bva_demo_last_year["QuestionText"] == list_keep_past[0]
        ]
        df_spe_last_lvs = self.df_lvs_demo[
            self.df_lvs_demo["QuestionText"] == list_keep_past[0]
        ]

        # Extract unique specializations
        specializations_this = self.extract_unique_classes(
            df_spe_this["AnswerText"].unique()
        )
        specializations_last_bva = self.extract_unique_classes(
            df_spe_last_bva["AnswerText"].unique()
        )
        specializations_last_lvs = self.extract_unique_classes(
            df_spe_last_lvs["AnswerText"].unique()
        )

        self.logger.info(
            f"Extracted {len(specializations_this)} unique specializations for this year"
        )
        self.logger.info(
            f"Extracted {len(specializations_last_bva)} unique specializations for {self.main_event_name} last year"
        )
        self.logger.info(
            f"Extracted {len(specializations_last_lvs)} unique specializations for {self.secondary_event_name} last year"
        )

        # Save specializations to JSON - use exact same structure as original
        specializations = {
            "this_year": specializations_this,
            "last_year_bva": specializations_last_bva,
            "last_year_lvs": specializations_last_lvs,
        }

        # Get output file name from config
        specializations_file = self.output_files.get("specializations", "specializations.json")
        with open(
            os.path.join(self.output_dir, "output", specializations_file), "w"
        ) as f:
            json.dump(specializations, f, indent=4)

    @staticmethod
    def extract_unique_classes(original_list):
        """
        Extract unique classes from a list of semicolon-separated values.

        Args:
            original_list: List of strings with semicolon-separated values

        Returns:
            List of unique classes
        """
        # Set to store unique classes
        unique_classes = set()

        # Process each item in the list
        for item in original_list:
            if not isinstance(item, str):
                continue

            # Split by semicolon and add each class to the set
            classes = item.split(";")
            for cls in classes:
                unique_classes.add(cls.strip())

        # Convert set back to a list for the final result
        return list(unique_classes)

    def extract_job_roles(self) -> None:
        """Extract and analyze job roles."""
        # Get job role question name from config
        job_role_question = self.config.get("job_role_question", "Job Role")
        
        # Extract job role data
        job_records_this = self.df_bva_demo_this_year[
            self.df_bva_demo_this_year["QuestionText"] == job_role_question
        ]
        job_records_last_bva = self.df_bva_demo_last_year[
            self.df_bva_demo_last_year["QuestionText"] == job_role_question
        ]
        job_records_last_lvs = self.df_lvs_demo[
            self.df_lvs_demo["QuestionText"] == job_role_question
        ]

        # Extract unique job roles
        job_roles_this = set(list(job_records_this["AnswerText"].unique()))
        job_roles_last_bva = set(list(job_records_last_bva["AnswerText"].unique()))
        job_roles_last_lvs = set(list(job_records_last_lvs["AnswerText"].unique()))

        # Combine all job roles
        all_job_roles = list(
            job_roles_this.union(job_roles_last_bva).union(job_roles_last_lvs)
        )

        self.logger.info(f"Extracted {len(all_job_roles)} unique job roles")

        # Get output file name from config
        job_roles_file = self.output_files.get("job_roles", "job_roles.json")
        with open(os.path.join(self.output_dir, "output", job_roles_file), "w") as f:
            json.dump(all_job_roles, f, indent=4)

    def find_demographics_for_returning_visitors(self) -> None:
        """Find demographic data for returning visitors."""
        # Get badge IDs for returning visitors - store these in class variables
        # so they can be used later in the combine_demographic_with_registration method
        self.list_badgeid_24_25_bva = list(self.df_bva_returning["BadgeId"].unique())
        self.list_badgeid_24_25_lva = list(self.df_lvs_returning["BadgeId"].unique())

        self.logger.info(f"{self.main_event_name} returning visitors: {len(self.list_badgeid_24_25_bva)}")
        self.logger.info(f"{self.secondary_event_name} returning visitors: {len(self.list_badgeid_24_25_lva)}")

        # Filter demographic data for returning visitors
        df_demo_24_25_bva = self.df_bva_demo_last_year[
            self.df_bva_demo_last_year["BadgeId"].isin(self.list_badgeid_24_25_bva)
        ]
        df_demo_24_25_lva = self.df_lvs_demo[
            self.df_lvs_demo["BadgeId"].isin(self.list_badgeid_24_25_lva)
        ]

        # Get output file names from config
        returning_demo_files = self.output_files.get("returning_demographic_data", {})
        
        # Save filtered demographic data
        df_demo_24_25_bva.to_csv(
            os.path.join(
                self.output_dir, "csv", 
                returning_demo_files.get("main_event", "Registration_demographicdata_returning_main.csv")
            ),
            index=False,
        )
        df_demo_24_25_lva.to_csv(
            os.path.join(
                self.output_dir, "csv", 
                returning_demo_files.get("secondary_event", "Registration_demographicdata_returning_secondary.csv")
            ),
            index=False,
        )

        self.logger.info(
            f"Found demographic data for {len(df_demo_24_25_bva)} returning {self.main_event_name} visitors"
        )
        self.logger.info(
            f"Found demographic data for {len(df_demo_24_25_lva)} returning {self.secondary_event_name} visitors"
        )

    def find_registration_with_demographic_data(self) -> None:
        """Find registration records that have demographic data."""
        # Get badge IDs with demographic data
        demo_badge_id_25 = list(self.df_bva_demo_this_year["BadgeId"].unique())
        demo_badge_id_24_bva = list(self.df_bva_demo_last_year["BadgeId"].unique())
        demo_badge_id_24_lva = list(self.df_lvs_demo["BadgeId"].unique())

        # Filter registration data for records with demographic data
        df_reg_25_wdemo_data = self.df_bva_this_year[
            self.df_bva_this_year["BadgeId"].isin(demo_badge_id_25)
        ]
        df_reg_24_wdemo_data_bva = self.df_bva_returning[
            self.df_bva_returning["BadgeId"].isin(demo_badge_id_24_bva)
        ]
        df_reg_24_wdemo_data_lva = self.df_lvs_returning[
            self.df_lvs_returning["BadgeId"].isin(demo_badge_id_24_lva)
        ]

        # Get output file names from config
        reg_with_demo_files = self.output_files.get("registration_with_demographic", {})

        # Save filtered registration data
        df_reg_25_wdemo_data.to_csv(
            os.path.join(
                self.output_dir,
                "output",
                reg_with_demo_files.get("this_year", "Registration_data_with_demographicdata_this_year.csv"),
            ),
            index=False,
        )
        df_reg_24_wdemo_data_bva.to_csv(
            os.path.join(
                self.output_dir,
                "output",
                reg_with_demo_files.get("last_year_main", "Registration_data_with_demographicdata_last_year_main.csv"),
            ),
            index=False,
        )
        df_reg_24_wdemo_data_lva.to_csv(
            os.path.join(
                self.output_dir,
                "output",
                reg_with_demo_files.get("last_year_secondary", "Registration_data_with_demographicdata_last_year_secondary.csv"),
            ),
            index=False,
        )

        self.logger.info(
            f"Found {len(df_reg_25_wdemo_data)} registration records with demographic data for this year"
        )
        self.logger.info(
            f"Found {len(df_reg_24_wdemo_data_bva)} registration records with demographic data for {self.main_event_name} last year"
        )
        self.logger.info(
            f"Found {len(df_reg_24_wdemo_data_lva)} registration records with demographic data for {self.secondary_event_name} last year"
        )
        # Get unique badge IDs again after concatenating with demo data
        demo_badge_id_25 = list(df_reg_25_wdemo_data["BadgeId"].unique())
        demo_badge_id_24_bva = list(df_reg_24_wdemo_data_bva["BadgeId"].unique())
        demo_badge_id_24_lva = list(df_reg_24_wdemo_data_lva["BadgeId"].unique())
        # Store for later use
        self.df_reg_25_wdemo_data = df_reg_25_wdemo_data
        self.df_reg_24_wdemo_data_bva = df_reg_24_wdemo_data_bva
        self.df_reg_24_wdemo_data_lva = df_reg_24_wdemo_data_lva
        self.demo_badge_id_25 = demo_badge_id_25
        self.demo_badge_id_24_bva = demo_badge_id_24_bva
        self.demo_badge_id_24_lva = demo_badge_id_24_lva

    def concatenate_qa_registration_data(self, df_registration, list_badge_id_events):
        """
        Concatenate Registration Values of a Participant.

        Args:
            df_registration: Pandas Dataframe with values of Participants
            list_badge_id_events: List with all BadgeId from which we have Geographic Data

        Returns:
            List of dictionaries containing concatenated registration data
        """

        def create_string_from_dict(data):
            keys_to_exclude = ["ShowRef", "BadgeId"]
            result = []

            for key, value in data.items():
                if key not in keys_to_exclude:
                    if value is not None:
                        result.append(f"{key}: {value}")
                    else:
                        result.append(f"{key}: no_data")

            return ", ".join(result)

        reg_data_list = []
        for badge_ID in list_badge_id_events:
            df = df_registration[df_registration["BadgeId"] == badge_ID]
            df_d = json.loads(df.to_json(orient="records"))
            reg_data = {}
            for row in df_d:
                ID = "_".join([row.get("ShowRef"), row.get("BadgeId")])
                reg_data[ID] = create_string_from_dict(row)
            if len(reg_data.keys()) > 0:
                reg_data_list.append(reg_data)

        self.logger.info(
            f"Created {len(reg_data_list)} concatenated registration data entries"
        )
        return reg_data_list

    def process_registration_data(self) -> None:
        """Process and save concatenated registration data."""
        # Define columns to keep
        valid_columns = [
            "Email",
            "Email_domain",
            "Company",
            "JobTitle",
            "Country",
            "BadgeType",
            "ShowRef",
            "BadgeId",
            "Source",
            "Days_since_registration",
        ]

        # Create valid column dataframes
        df_reg_25_valid_columns = self.df_reg_25_wdemo_data[valid_columns]
        df_reg_24_25_bva_valid_columns = self.df_reg_24_wdemo_data_bva[valid_columns]
        df_reg_24_25_lva_valid_columns = self.df_reg_24_wdemo_data_lva[valid_columns]

        # Create concatenated registration data
        reg_data_list_this_year = self.concatenate_qa_registration_data(
            df_reg_25_valid_columns, self.demo_badge_id_25
        )
        reg_data_list_past_year_bva = self.concatenate_qa_registration_data(
            df_reg_24_25_bva_valid_columns, self.demo_badge_id_24_bva
        )
        reg_data_list_past_year_lva = self.concatenate_qa_registration_data(
            df_reg_24_25_lva_valid_columns, self.demo_badge_id_24_lva
        )

        # Get output file names from config
        concatenated_reg_files = self.output_files.get("concatenated_registration_data", {})

        # Save concatenated registration data
        with open(
            os.path.join(
                self.output_dir, "output", 
                concatenated_reg_files.get("this_year", "main_registration_data_this_year.json")
            ),
            "w",
        ) as f:
            json.dump(reg_data_list_this_year, f, indent=4)
        with open(
            os.path.join(
                self.output_dir, "output", 
                concatenated_reg_files.get("past_year_main", "main_registration_data_past_year.json")
            ),
            "w",
        ) as f:
            json.dump(reg_data_list_past_year_bva, f, indent=4)
        with open(
            os.path.join(
                self.output_dir, "output", 
                concatenated_reg_files.get("past_year_secondary", "secondary_registration_data_past_year.json")
            ),
            "w",
        ) as f:
            json.dump(reg_data_list_past_year_lva, f, indent=4)

        self.logger.info("Saved concatenated registration data to JSON files")

    def concatenate_qa_demographic_data(
        self, df_demographic, list_badge_id_events, list_keep
    ):
        """
        Concatenate Questions and Answers of a Participant.

        Args:
            df_demographic: Pandas Dataframe with values of Participants
            list_badge_id_events: List with all BadgeId from which we have Geographic Data
            list_keep: List of questions that should be kept

        Returns:
            List of dictionaries containing demographic data
        """
        demo_data = []

        for badge_ID in list_badge_id_events:
            df = df_demographic[df_demographic["BadgeId"] == badge_ID]
            df_d = json.loads(df.to_json(orient="records"))
            qa = {badge_ID: {}}
            for row in df_d:
                question = row.get("QuestionText")
                answer = row.get("AnswerText")

                if not question or not answer:
                    logging.warning(
                        f"Issue with records ID {badge_ID} NO ANSWER/QUESTION in demographic data"
                    )
                    continue
                if question in list_keep:
                    qq = question.lower().replace(" ", "_")
                    qa[badge_ID][qq] = answer
            if len(qa[badge_ID].keys()) > 0:
                demo_data.append(qa)
            else:
                logging.warning(f"No demo data for {badge_ID}")

        self.logger.info(f"Created {len(demo_data)} demographic data entries")
        return demo_data

    def process_demographic_data(self) -> None:
        """Process and save demographic data."""
        # Get the questions to keep from config
        questions_to_keep = self.config.get("questions_to_keep", {})
        list_keep_this = questions_to_keep.get("current", [])
        list_keep_past = questions_to_keep.get("past", [])

        # Process demographic data
        demo_data_this = self.concatenate_qa_demographic_data(
            self.df_bva_demo_this_year, self.demo_badge_id_25, list_keep_this
        )
        demo_data_last_bva = self.concatenate_qa_demographic_data(
            self.df_bva_demo_last_year, self.demo_badge_id_24_bva, list_keep_past
        )
        demo_data_last_lva = self.concatenate_qa_demographic_data(
            self.df_lvs_demo, self.demo_badge_id_24_lva, list_keep_past
        )

        # Get output file names from config
        demo_output_files = self.output_files.get("processed_demographic_data", {})

        # Save demographic data
        with open(
            os.path.join(self.output_dir, "output", 
                        demo_output_files.get("this_year", "demographic_data_this.json")), "w"
        ) as f:
            json.dump(demo_data_this, f, indent=4)
        with open(
            os.path.join(self.output_dir, "output", 
                        demo_output_files.get("last_year_main", "demographic_data_last_main.json")),
            "w",
        ) as f:
            json.dump(demo_data_last_bva, f, indent=4)
        with open(
            os.path.join(self.output_dir, "output", 
                        demo_output_files.get("last_year_secondary", "demographic_data_last_secondary.json")),
            "w",
        ) as f:
            json.dump(demo_data_last_lva, f, indent=4)

        self.logger.info("Saved demographic data to JSON files")

    def process_with_demographics(self) -> None:
        """Execute the workflow for processing demographic data."""
        self.logger.info("Starting demographic data processing workflow")

        self.save_demographic_data()
        self.extract_specialization_data()
        self.extract_job_roles()
        self.find_demographics_for_returning_visitors()
        self.find_registration_with_demographic_data()
        self.process_registration_data()
        self.process_demographic_data()

        self.logger.info("Demographic data processing workflow completed successfully")

    def create_col_placeholders(self, df, list_to_keep):
        """
        Create placeholder columns in the dataframe for demographic data.

        Args:
            df: DataFrame to add columns to
            list_to_keep: List of column names to add

        Returns:
            DataFrame with placeholder columns added
        """
        df_copy = df.copy()
        
        # Only create job_role column for veterinary events
        if hasattr(self, '_vet_specific_active') and self._vet_specific_active:
            if "job_role" not in df_copy.columns:
                df_copy["job_role"] = "NA"
        
        # Add demographic columns
        for col in list_to_keep:
            qq = col.lower().replace(" ", "_")
            if qq not in df_copy.columns:
                df_copy[qq] = "NA"
        
        return df_copy

    def create_democols_in_registration_data(self, df, demo_data, list_keep):
        """
        Populate columns with demographic data values.

        Args:
            df: DataFrame to populate
            demo_data: List of dictionaries containing demographic data
            list_keep: List of columns to populate

        Returns:
            DataFrame with populated demographic data
        """
        # Create new columns with placeholders
        df = self.create_col_placeholders(df, list_keep)

        # Populate with demo data
        for reg in demo_data:
            badgeid = list(reg.keys())[0]
            list_keys_this_reg = list(reg[badgeid].keys())
            for col in list_keys_this_reg:
                df.loc[df["BadgeId"] == badgeid, col] = reg[badgeid][col]

        self.logger.info(
            f"Populated demographic data columns for {len(demo_data)} records"
        )
        return df

    def process_job_roles(self, df):
        """
        Process and standardize job roles using rules and fuzzy matching.
        This is ONLY for veterinary events. Generic events skip this step entirely.

        Args:
            df: DataFrame containing job role information

        Returns:
            DataFrame (unchanged for generic events)
        """
        # Check if this is a veterinary event with vet-specific functions
        if not (hasattr(self, '_vet_specific_active') and self._vet_specific_active):
            # For generic events (like ECOMM), skip job role processing entirely
            self.logger.info("Skipping job role processing for generic event (not a veterinary event)")
            return df
        
        # If we get here, this is a vet event and should have been overridden by vet-specific functions
        self.logger.warning("process_job_roles called for vet event but vet-specific functions not properly applied")
        return df

    # Replace the fill_missing_practice_types method in registration_processor.py:

    def fill_missing_practice_types(self, df, practices, column):
        """
        Fill missing practice types using fuzzy matching with company names.
        This is ONLY for veterinary events. Generic events skip this entirely.

        Args:
            df: DataFrame with potentially missing practice types
            practices: DataFrame with company names and practice types
            column: Name of the column containing practice types

        Returns:
            DataFrame (unchanged for generic events)
        """
        # Check if this is a veterinary event with vet-specific functions
        if not (hasattr(self, '_vet_specific_active') and self._vet_specific_active):
            # For generic events (like ECOMM), skip practice type filling entirely
            self.logger.info("Skipping practice type filling for generic event (not a veterinary event)")
            return df
        
        # If we get here, this is a vet event and should have been overridden by vet-specific functions
        self.logger.warning("fill_missing_practice_types called for vet event but vet-specific functions not properly applied")
        return df

    def combine_demographic_with_registration(self):
        """Process and combine demographic data with registration data."""
        self.logger.info("Starting to combine demographic data with registration data")

        # Get the questions to keep from config
        questions_to_keep = self.config.get("questions_to_keep", {})
        list_keep_this = questions_to_keep.get("current", [])
        list_keep_past = questions_to_keep.get("past", [])

        # Get badge history column names from config
        badge_history_cols = self.config.get("badge_history_columns", {})
        main_col = badge_history_cols.get("main_event", "BadgeId_last_year_bva")
        secondary_col = badge_history_cols.get("secondary_event", "BadgeId_last_year_lva")

        # Create dataframes with valid columns for registration data
        valid_columns = [
            "Email",
            "Email_domain",
            "Company",
            "JobTitle",
            "Country",
            "BadgeType",
            "ShowRef",
            "BadgeId",
            "Source",
            "Days_since_registration",
            "assist_year_before",
            main_col,
            secondary_col,
        ]

        # Ensure we have all the columns in the dataframe
        for col in valid_columns:
            if col not in self.df_reg_25_wdemo_data.columns:
                self.logger.warning(f"Column {col} not in df_reg_25_wdemo_data")
                if col in [main_col, secondary_col, "assist_year_before"]:
                    self.logger.info(f"Adding missing column {col}")
                    self.df_reg_25_wdemo_data[col] = "NA"

        # Now select only the columns we need
        available_columns = [
            col for col in valid_columns if col in self.df_reg_25_wdemo_data.columns
        ]
        df_reg_25_valid_columns = self.df_reg_25_wdemo_data[available_columns].copy()

        # For past year data, we don't need the badge history columns
        past_valid_columns = [
            col
            for col in valid_columns
            if col not in [main_col, secondary_col, "assist_year_before"]
        ]

        # Create valid column dataframes for returning visitors
        # Use the returning visitors dataframes directly instead of the filtered ones with demo data
        df_reg_24_25_bva_valid_columns = self.df_bva_returning[
            past_valid_columns
        ].copy()
        df_reg_24_25_lva_valid_columns = self.df_lvs_returning[
            past_valid_columns
        ].copy()

        # Filter current year to only include badges with demo data
        df_reg_25_valid_columns = df_reg_25_valid_columns[
            df_reg_25_valid_columns["BadgeId"].isin(self.demo_badge_id_25)
        ]

        # Log initial counts before any filtering
        self.logger.info(
            f"Initial counts - {self.main_event_name} returning: {len(self.df_bva_returning)}, {self.secondary_event_name} returning: {len(self.df_lvs_returning)}"
        )

        # Log the lengths after filtering
        self.logger.info(
            f"After filtering: current year: {len(df_reg_25_valid_columns)}, {self.main_event_name} last year: {len(df_reg_24_25_bva_valid_columns)}, {self.secondary_event_name} last year: {len(df_reg_24_25_lva_valid_columns)}"
        )

        # Process demographic data
        demo_data_this = self.concatenate_qa_demographic_data(
            self.df_bva_demo_this_year, self.demo_badge_id_25, list_keep_this
        )

        # For past years, we still need demographic data, but we'll apply it to all returning visitors
        # not just those with demo data from previous year
        demo_data_last_bva = self.concatenate_qa_demographic_data(
            self.df_bva_demo_last_year, self.demo_badge_id_24_bva, list_keep_past
        )
        demo_data_last_lva = self.concatenate_qa_demographic_data(
            self.df_lvs_demo, self.demo_badge_id_24_lva, list_keep_past
        )

        self.logger.info(
            f"Processed demographic data. Created demographic data for {len(demo_data_this)} this year, {len(demo_data_last_bva)} last year {self.main_event_name}, and {len(demo_data_last_lva)} last year {self.secondary_event_name}"
        )
        # Create combined registration and demographic data
        df_reg_demo_this = self.create_democols_in_registration_data(
            df_reg_25_valid_columns, demo_data_this, list_keep_this
        )
        df_reg_demo_last_bva = self.create_democols_in_registration_data(
            df_reg_24_25_bva_valid_columns, demo_data_last_bva, list_keep_past
        )
        df_reg_demo_last_lva = self.create_democols_in_registration_data(
            df_reg_24_25_lva_valid_columns, demo_data_last_lva, list_keep_past
        )

        # Process job roles
        df_reg_demo_this = self.process_job_roles(df_reg_demo_this)
        df_reg_demo_last_bva = self.process_job_roles(df_reg_demo_last_bva)
        df_reg_demo_last_lva = self.process_job_roles(df_reg_demo_last_lva)
        self.logger.info("Processed job roles for all demographic data entries")
        self.logger.info("Length of df_reg_demo_this: " + str(len(df_reg_demo_this)))
        self.logger.info(
            "Length of df_reg_demo_last_bva: " + str(len(df_reg_demo_last_bva))
        )
        self.logger.info(
            "Length of df_reg_demo_last_lva: " + str(len(df_reg_demo_last_lva))
        )

        # Check for badge history columns
        if main_col in df_reg_demo_this.columns:
            self.logger.info(f"{main_col} column exists in final dataframe")
            self.logger.info(
                f"Number of unique {main_col} values: {len(df_reg_demo_this[main_col].unique())}"
            )
        else:
            self.logger.warning(
                f"{main_col} column missing from final dataframe"
            )

        if secondary_col in df_reg_demo_this.columns:
            self.logger.info(f"{secondary_col} column exists in final dataframe")
            self.logger.info(
                f"Number of unique {secondary_col} values: {len(df_reg_demo_this[secondary_col].unique())}"
            )
        else:
            self.logger.warning(
                f"{secondary_col} column missing from final dataframe"
            )

        # Only process practices for veterinary events
        if hasattr(self, '_vet_specific_active') and self._vet_specific_active:
            # Load practice data for filling missing practice types using vet-specific logic
            practices_file = self.config.get("input_files", {}).get("practices", "")
            if practices_file and os.path.exists(practices_file):
                try:
                    practices = pd.read_csv(practices_file)
                    
                    # Call vet-specific practice filling function
                    df_reg_demo_this, df_reg_demo_last_bva, df_reg_demo_last_lva = self.fill_event_specific_practice_types(
                        df_reg_demo_this, df_reg_demo_last_bva, df_reg_demo_last_lva, practices
                    )

                    self.logger.info("Filled missing practice types using vet-specific practices data")
                except Exception as e:
                    self.logger.error(f"Error loading or processing practices data: {e}")
            else:
                if practices_file:
                    self.logger.warning(f"Practices file '{practices_file}' not found. Skipping practice type filling.")
                else:
                    self.logger.info("No practices file configured for vet event. Skipping practice type filling.")
        else:
            # For generic events (like ECOMM), skip practice processing entirely
            self.logger.info("Skipping all practice type processing for generic event (not a veterinary event)")

        # Get output file names from config
        combined_output_files = self.output_files.get("combined_demographic_registration", {})

        # Save combined data
        df_reg_demo_this.to_csv(
            os.path.join(self.output_dir, "output", 
                        combined_output_files.get("this_year", "df_reg_demo_this.csv")), index=False
        )
        df_reg_demo_last_bva.to_csv(
            os.path.join(self.output_dir, "output", 
                        combined_output_files.get("last_year_main", "df_reg_demo_last_bva.csv")),
            index=False,
        )
        df_reg_demo_last_lva.to_csv(
            os.path.join(self.output_dir, "output", 
                        combined_output_files.get("last_year_secondary", "df_reg_demo_last_lva.csv")),
            index=False,
        )
        self.logger.info(
            "Saved combined demographic and registration data to CSV files"
        )

        # Store for potential further use - keep original variable names for backward compatibility
        self.df_reg_demo_this = df_reg_demo_this
        self.df_reg_demo_last_bva = df_reg_demo_last_bva
        self.df_reg_demo_last_lva = df_reg_demo_last_lva

        self.logger.info("Finished combining demographic with registration data")

        # Return statistics
        return {
            "this_year": len(df_reg_demo_this),
            "last_year_bva": len(df_reg_demo_last_bva),
            "last_year_lva": len(df_reg_demo_last_lva),
        }

    def fill_event_specific_practice_types(self, df_reg_demo_this, df_reg_demo_last_bva, df_reg_demo_last_lva, practices):
        """
        Fill missing practice types using event-specific logic.
        This method can be overridden by event-specific modules (e.g., veterinary events).
        
        The GENERIC version uses configuration to determine column names and behavior.

        Args:
            df_reg_demo_this: Current year combined dataframe
            df_reg_demo_last_bva: Last year main event dataframe
            df_reg_demo_last_lva: Last year secondary event dataframe
            practices: Practices reference dataframe

        Returns:
            Tuple of (df_reg_demo_this, df_reg_demo_last_bva, df_reg_demo_last_lva)
        """
        self.logger.info(f"Filling practice types using {'vet-specific' if hasattr(self, '_vet_specific_active') else 'generic'} event logic")
        
        # Use practice_type_columns from config for generic behavior
        practice_columns = self.config.get("practice_type_columns", {})
        this_year_col = practice_columns.get("current", "specialization_current")
        past_year_col = practice_columns.get("past", "specialization_past")

        self.logger.info(f"Using practice type columns - current: {this_year_col}, past: {past_year_col}")

        # Fill missing practice types using generic logic
        df_reg_demo_this = self.fill_missing_practice_types(
            df_reg_demo_this, practices, column=this_year_col
        )
        df_reg_demo_last_bva = self.fill_missing_practice_types(
            df_reg_demo_last_bva, practices, column=past_year_col
        )
        df_reg_demo_last_lva = self.fill_missing_practice_types(
            df_reg_demo_last_lva, practices, column=past_year_col
        )
        
        return df_reg_demo_this, df_reg_demo_last_bva, df_reg_demo_last_lva


    def process(self) -> None:
        """Execute the full data processing workflow."""
        self.logger.info("Starting data processing workflow")

        self.load_data()
        self.save_initial_data()
        self.preprocess_data()
        self.split_and_process_bva_data()
        self.identify_returning_visitors()
        self.add_badge_history()
        self.calculate_event_dates()
        self.flag_all_returning_visitors()
        self.select_and_save_final_data()
        self.process_with_demographics()
        self.combine_demographic_with_registration()

        self.logger.info("Data processing workflow completed successfully")