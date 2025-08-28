import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
import logging
import concurrent.futures
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# Optional imports (for LangChain approach)
try:
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_openai import ChatOpenAI, AzureChatOpenAI, AzureOpenAI
    from azure.ai.inference import ChatCompletionsClient
    from azure.core.credentials import AzureKeyCredential
    from dotenv import dotenv_values

    has_langchain = True
except ImportError:
    has_langchain = False
    print("LangChain not available. Will use rule-based filtering only.")


class SessionRecommendationProcessor:
    """
    Process and generate session recommendations for visitors.
    """

    def __init__(self, config):
        """
        Initialize the session recommendation processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = config.get("output_dir", "data/bva")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "recommendations"), exist_ok=True)

        # Neo4j connection parameters
        self.uri = config["neo4j"]["uri"]
        self.username = config["neo4j"]["username"]
        self.password = config["neo4j"]["password"]

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize sentence transformer model
        self.model_name = "all-MiniLM-L6-v2"
        self.model = None  # Lazy initialization

        # Initialize statistics
        self.statistics = {
            "total_visitors_processed": 0,
            "visitors_with_recommendations": 0,
            "total_recommendations_generated": 0,
            "unique_recommended_sessions": 0,
            "processing_time": 0,
            "errors": 0,
            "error_details": [],
        }

        # Define role groups (for rule processing)
        self.vet_roles = [
            "Vet/Vet Surgeon",
            "Assistant Vet",
            "Vet/Owner",
            "Clinical or other Director",
            "Locum Vet",
            "Academic",
        ]

        self.nurse_roles = ["Head Nurse/Senior Nurse", "Vet Nurse", "Locum RVN"]

        self.business_roles = ["Practice Manager", "Practice Partner/Owner"]
        # Other roles can attend any session
        self.other_roles = ["Student", "Receptionist", "Other (please specify)"]

        # Initialize caches
        self._this_year_sessions_cache = None
        self._visitor_cache = {}
        self._similar_visitors_cache = {}

        # Load business rules
        self.business_rules = """
        1.) if visitor what_type_does_your_practice_specialise_in contains "equine" or "mixed", you can't propose session on stream "exotics", "feline", "exotic animal", "farm", "small animal"
        2.) if visitor what_type_does_your_practice_specialise_in contains "small animal", you can't propose session on stream "equine", "farm animal", "farm", "large animal"
        3.) if job_role in VET_ROLES session.stream cant be "nursing"
        4.) if job_role in NURSE_ROLES you only recommend sessions in stream "nursing", "wellbeing", "welfare"
        5.) rule 1 and 2 are mutually exclusive and apply first then apply 3 and 4
        """

        # Rules configuration
        self.rules_config = {
            "equine_mixed_exclusions": [
                "exotics",
                "feline",
                "exotic animal",
                "farm",
                "small animal",
            ],
            "small_animal_exclusions": [
                "equine",
                "farm animal",
                "farm",
                "large animal",
            ],
            "vet_exclusions": ["nursing"],
            "nurse_streams": ["nursing", "wellbeing", "welfare"],
            "rule_priority": ["practice_type", "role"],  # Order of rule application
        }

    def _get_visitors_to_process(self, create_only_new):
        """
        Get list of visitors to process based on create_only_new flag.

        Args:
            create_only_new: If True, only get visitors without recommendations

        Returns:
            List of badge IDs to process
        """
        try:
            with GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            ) as driver:
                with driver.session() as session:
                    if create_only_new:
                        # Only get visitors without recommendations
                        query = """
                        MATCH (v:Visitor_this_year)
                        WHERE v.has_recommendation IS NULL OR v.has_recommendation = "0"
                        RETURN v.BadgeId as badge_id
                        """
                    else:
                        # Get all visitors
                        query = """
                        MATCH (v:Visitor_this_year)
                        RETURN v.BadgeId as badge_id
                        """

                    result = session.run(query)
                    badge_ids = [record["badge_id"] for record in result]

                    self.logger.info(f"Found {len(badge_ids)} visitors to process")
                    return badge_ids

        except Exception as e:
            self.logger.error(
                f"Error getting visitors to process: {str(e)}", exc_info=True
            )
            return []

    def _update_visitor_recommendations(self, recommendations_data):
        """
        Update visitors with has_recommendation flag and create IS_RECOMMENDED relationships.

        Args:
            recommendations_data: List of recommendation results with visitor and session data
        """
        try:
            with GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            ) as driver:
                with driver.session() as session:
                    for rec_data in recommendations_data:
                        badge_id = rec_data["metadata"]["badge_id"]
                        filtered_recs = rec_data.get("filtered_recommendations", [])

                        if len(filtered_recs) > 0:
                            # Update has_recommendation flag
                            update_query = """
                            MATCH (v:Visitor_this_year {BadgeId: $badge_id})
                            SET v.has_recommendation = "1"
                            RETURN v
                            """
                            session.run(update_query, badge_id=badge_id)

                            # Create IS_RECOMMENDED relationships
                            for rec in filtered_recs:
                                session_id = rec.get("session_id")
                                if session_id:
                                    rel_query = """
                                    MATCH (v:Visitor_this_year {BadgeId: $badge_id})
                                    MATCH (s:Sessions_this_year {session_id: $session_id})
                                    MERGE (v)-[r:IS_RECOMMENDED]->(s)
                                    RETURN r
                                    """
                                    session.run(
                                        rel_query,
                                        badge_id=badge_id,
                                        session_id=session_id,
                                    )

                            self.logger.info(
                                f"Updated visitor {badge_id} with {len(filtered_recs)} recommendations"
                            )

        except Exception as e:
            self.logger.error(
                f"Error updating visitor recommendations: {str(e)}", exc_info=True
            )

    def json_to_dataframe(self, json_file_path: str) -> pd.DataFrame:
        """
        Convert JSON data to a pandas DataFrame with one row per filtered recommendation,
        handling both string and dictionary filtered recommendations.
        
        Args:
            json_file_path: Path to the JSON file
            
        Returns:
            pandas DataFrame with visitor keys + session keys + metadata keys
        """
        # Read the JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        rows = []
        
        # Access the recommendations list
        recommendations = data.get('recommendations', [])
        
        # Iterate through each recommendation
        for rec in recommendations:
            visitor = rec.get('visitor', {})  # Dictionary with visitor info
            filtered_recs = rec.get('filtered_recommendations', [])  # List of dictionaries or strings
            rec_metadata = rec.get('metadata', {})  # Dictionary with recommendation metadata
            
            # For each filtered recommendation, create a row
            for filtered_rec in filtered_recs:
                row = {}
                
                # Add visitor data
                for key, value in visitor.items():
                    row[f"visitor_{key}"] = value
                
                # Add filtered recommendation data
                if isinstance(filtered_rec, dict):
                    # If it's a dictionary, add all key-value pairs
                    for key, value in filtered_rec.items():
                        row[f"session_{key}"] = value
                else:
                    # If it's a string or other simple type, assume it's a session_id
                    row["session_id"] = filtered_rec
                
                # Add recommendation metadata
                for key, value in rec_metadata.items():
                    row[f"metadata_{key}"] = value
                
                rows.append(row)
        
        # Create DataFrame from the list of rows
        df = pd.DataFrame(rows)
        
        return df
    
    def process(self, create_only_new=True):
        """
        Run the session recommendation processor.

        Args:
            create_only_new: If True, only process new recommendations if the output file doesn't exist
        """
        start_time = time.time()
        self.logger.info("Starting session recommendation processing")

        try:
            # Get visitors to process based on create_only_new flag
            badge_ids = self._get_visitors_to_process(create_only_new)

            if not badge_ids or len(badge_ids) == 0:
                self.logger.info("No visitors to process for recommendations.")
                return

            # Load visitor data
            visitor_data = self._load_visitor_data()
            if visitor_data is None or len(visitor_data) == 0:
                self.logger.error(
                    "No visitor data found. Cannot generate recommendations."
                )
                return

            # Filter visitor data to only include the badge IDs we want to process
            visitor_data = visitor_data[visitor_data["BadgeId"].isin(badge_ids)]

            self.logger.info(f"Processing {len(visitor_data)} visitors")
            self.statistics["total_visitors_processed"] = len(visitor_data)

            # Get unique badge IDs
            badge_ids = visitor_data["BadgeId"].unique()
            self.logger.info(f"Found {len(badge_ids)} unique badge IDs")

            # Create timestamp for output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.output_dir,
                "recommendations",
                f"visitor_recommendations_{timestamp}.json",
            )

            # Check if file already exists and if we should only create new ones
            if os.path.exists(output_file) and create_only_new:
                self.logger.info(
                    f"Output file {output_file} already exists. Skipping processing."
                )
                return

            # Get configuration settings for recommendations
            min_score = self.config.get("recommendation", {}).get(
                "min_similarity_score", 0.3
            )
            max_recommendations = self.config.get("recommendation", {}).get(
                "max_recommendations", 10
            )
            use_langchain = self.config.get("recommendation", {}).get(
                "use_langchain", False
            )

            self.logger.info(
                f"Using configuration: min_score={min_score}, max_recommendations={max_recommendations}, use_langchain={use_langchain}"
            )

            # Generate recommendations for each visitor
            all_recommendations = []
            errors = 0
            successful = 0

            # Initialize lazily
            self._init_model()

            for badge_id in badge_ids:
                try:
                    self.logger.info(
                        f"Processing recommendations for visitor {badge_id}"
                    )

                    # Get visitor data for this badge ID
                    visitor_rows = visitor_data[visitor_data["BadgeId"] == badge_id]
                    if visitor_rows.empty:
                        continue

                    visitor = visitor_rows.iloc[0].to_dict()

                    # Generate recommendations
                    result = self.get_recommendations_and_filter(
                        badge_id=badge_id,
                        min_score=min_score,
                        max_recommendations=max_recommendations,
                        visitor_data=visitor,
                        use_langchain=use_langchain,
                    )

                    # Add to results if recommendations were generated
                    if (
                        result
                        and "filtered_recommendations" in result
                        and len(result["filtered_recommendations"]) > 0
                    ):
                        all_recommendations.append(result)
                        successful += 1

                except Exception as e:
                    self.logger.error(
                        f"Error generating recommendations for visitor {badge_id}: {str(e)}",
                        exc_info=True,
                    )
                    errors += 1
                    self.statistics["error_details"].append(
                        f"Badge ID {badge_id}: {str(e)}"
                    )

            # Update statistics
            self.statistics["visitors_with_recommendations"] = successful
            self.statistics["errors"] = errors

            # Calculate unique recommended sessions
            unique_sessions = set()
            total_recommendations = 0
            for result in all_recommendations:
                if "filtered_recommendations" in result:
                    total_recommendations += len(result["filtered_recommendations"])
                    for rec in result["filtered_recommendations"]:
                        if "session_id" in rec:
                            unique_sessions.add(rec["session_id"])

            self.statistics["total_recommendations_generated"] = total_recommendations
            self.statistics["unique_recommended_sessions"] = len(unique_sessions)

            # Save all recommendations to a JSON file
            output_data = {
                "recommendations": all_recommendations,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_visitors": len(badge_ids),
                    "successful_recommendations": successful,
                    "errors": errors,
                    "total_recommendations": total_recommendations,
                    "unique_sessions_recommended": len(unique_sessions),
                },
            }

            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2, default=str)

            self.logger.info(f"Saved recommendations to {output_file}")
            self.logger.info(
                f"Generated recommendations for {successful} visitors with {errors} errors"
            )
            # Convert to DataFrame and save as CSV
            df = self.json_to_dataframe(output_file)
            csv_file = output_file.replace(".json", ".csv")
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Saved recommendations DataFrame to {csv_file}")
            
            # Update Neo4j with recommendations
            if successful > 0:
                self.logger.info("Updating Neo4j with recommendation data")
                self._update_visitor_recommendations(all_recommendations)
                self.logger.info("Completed Neo4j updates")

            # Calculate processing time
            self.statistics["processing_time"] = time.time() - start_time
            self.logger.info(
                f"Completed session recommendation processing in {self.statistics['processing_time']:.2f} seconds"
            )

        except Exception as e:
            self.logger.error(
                f"Error in session recommendation processing: {str(e)}", exc_info=True
            )
            self.statistics["errors"] += 1
            self.statistics["error_details"].append(f"General error: {str(e)}")
            self.statistics["processing_time"] = time.time() - start_time

    def _load_visitor_data(self):
        """
        Load visitor data from the configured output directory.

        Returns:
            DataFrame with visitor data or None if not found
        """
        try:
            # Check for visitor data file
            visitor_file = os.path.join(
                self.config.get("output_dir", "data/bva"),
                "output",
                "df_reg_demo_this.csv",
            )

            if not os.path.exists(visitor_file):
                self.logger.error(f"Visitor data file {visitor_file} not found")
                return None

            visitor_data = pd.read_csv(visitor_file)
            return visitor_data

        except Exception as e:
            self.logger.error(f"Error loading visitor data: {str(e)}", exc_info=True)
            return None

    def _init_model(self):
        """Initialize the sentence transformer model if not already initialized."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.logger.info(
                    f"Initialized sentence transformer model {self.model_name}"
                )
            except Exception as e:
                self.logger.error(
                    f"Error initializing sentence transformer model: {str(e)}",
                    exc_info=True,
                )
                raise

    def clear_caches(self):
        """Clear all caches."""
        self._this_year_sessions_cache = None
        self._visitor_cache = {}
        self._similar_visitors_cache = {}

    def get_this_year_sessions(self, tx):
        """
        Get all sessions for this year with their embeddings.
        Uses caching for better performance.
        """
        if self._this_year_sessions_cache is not None:
            return self._this_year_sessions_cache

        # Query to get this year's sessions with embeddings
        query = """
        MATCH (s:Sessions_this_year)
        WHERE s.embedding IS NOT NULL
        RETURN s.session_id as session_id, 
               s.title as title, 
               s.stream as stream, 
               s.synopsis_stripped as synopsis_stripped,
               s.theatre__name as theatre__name,
               s.embedding as embedding,
               s.date as date,
               s.start_time as start_time,
               s.end_time as end_time,
               s.sponsored_by as sponsored_by,
               s.sponsored_session as sponsored_session
        """

        results = tx.run(query).data()

        # Process results and cache them
        sessions = {}
        for r in results:
            embedding = np.array(json.loads(r["embedding"])) if r["embedding"] else None

            if embedding is not None:
                sessions[r["session_id"]] = {
                    "session_id": r["session_id"],
                    "title": r["title"],
                    "stream": r["stream"],
                    "synopsis_stripped": r["synopsis_stripped"],
                    "theatre__name": r["theatre__name"],
                    "embedding": embedding,
                    "date": r["date"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "sponsored_by": r["sponsored_by"],
                    "sponsored_session": r["sponsored_session"],
                }

        self._this_year_sessions_cache = sessions
        return sessions

    def get_visitor_info(self, tx, visitor_id):
        """
        Get visitor information with caching.
        """
        if visitor_id in self._visitor_cache:
            return self._visitor_cache[visitor_id]

        visitor_query = """
        MATCH (v:Visitor_this_year {BadgeId: $visitor_id})
        RETURN v
        """
        visitor_data = tx.run(visitor_query, visitor_id=visitor_id).single()

        if not visitor_data:
            return None

        visitor = visitor_data["v"]
        assisted = visitor.get("assist_year_before", "0")

        self._visitor_cache[visitor_id] = {"visitor": visitor, "assisted": assisted}
        return self._visitor_cache[visitor_id]

    def get_past_sessions(self, tx, visitor_id):
        """
        Get sessions the visitor attended last year.
        Optimized with a more efficient query.
        """
        # Single query combining both visitor types
        query_past = """
        MATCH (v:Visitor_this_year {BadgeId: $visitor_id})-[:Same_Visitor]->(vp)-[:attended_session]->(sp:Sessions_past_year)
        WHERE (vp:Visitor_last_year_bva OR vp:Visitor_last_year_lva)
        RETURN DISTINCT sp.session_id as session_id, sp.embedding as embedding
        """

        results = tx.run(query_past, visitor_id=visitor_id).data()

        # Process embeddings
        sessions = []
        for r in results:
            embedding = np.array(json.loads(r["embedding"])) if r["embedding"] else None
            if embedding is not None:
                sessions.append({"session_id": r["session_id"], "embedding": embedding})

        return sessions

    def find_similar_visitors_batch(self, tx, visitor, num_similar_visitors=3):
        """
        Find similar visitors with batch processing.
        Uses a more efficient query and caching.
        """
        visitor_id = visitor["BadgeId"]

        # Check cache first
        if visitor_id in self._similar_visitors_cache:
            return self._similar_visitors_cache[visitor_id]

        # Get all visitors with sessions in one query
        # Note: We always look at ALL visitors who attended last year, not just new ones
        query = """
        MATCH (v:Visitor_this_year)
        WHERE v.assist_year_before = '1' AND v.BadgeId <> $visitor_id
        // Pre-filter to avoid processing all visitors
        WITH v, 
             CASE WHEN v.job_role = $job_role THEN 1 ELSE 0 END + 
             CASE WHEN v.what_type_does_your_practice_specialise_in = $practice_type THEN 1 ELSE 0 END +
             CASE WHEN v.organisation_type = $org_type THEN 1 ELSE 0 END +
             CASE WHEN v.Country = $country THEN 1 ELSE 0 END AS base_similarity
        // Only process those with some similarity
        WHERE base_similarity > 0
        // Check if they have attended sessions (to save processing visitors without sessions)
        MATCH (v)-[:Same_Visitor]->(vp)-[:attended_session]->(sp:Sessions_past_year)
        WHERE (vp:Visitor_last_year_bva OR vp:Visitor_last_year_lva)
        WITH v, base_similarity, COUNT(DISTINCT sp) AS session_count
        WHERE session_count > 0
        RETURN v, base_similarity
        ORDER BY base_similarity DESC, session_count DESC
        LIMIT 20
        """

        visitors_data = tx.run(
            query,
            visitor_id=visitor_id,
            job_role=visitor.get("job_role", ""),
            practice_type=visitor.get("what_type_does_your_practice_specialise_in", ""),
            org_type=visitor.get("organisation_type", ""),
            country=visitor.get("Country", ""),
        ).data()

        # If we can't find enough similar visitors with the pre-filtering,
        # try a more general query
        if len(visitors_data) < num_similar_visitors:
            query = """
            MATCH (v:Visitor_this_year)
            WHERE v.assist_year_before = '1' AND v.BadgeId <> $visitor_id
            // Check if they have attended sessions
            MATCH (v)-[:Same_Visitor]->(vp)-[:attended_session]->(sp:Sessions_past_year)
            WHERE (vp:Visitor_last_year_bva OR vp:Visitor_last_year_lva)
            WITH v, COUNT(DISTINCT sp) AS session_count
            WHERE session_count > 0
            RETURN v, 0 AS base_similarity
            ORDER BY session_count DESC
            LIMIT 20
            """
            visitors_data = tx.run(query, visitor_id=visitor_id).data()

        # Extract visitor features for comparison
        def get_visitor_features(v):
            attributes = [
                v.get("what_type_does_your_practice_specialise_in", ""),
                v.get("job_role", ""),
                v.get("organisation_type", ""),
                v.get("JobTitle", ""),
                v.get("Country", ""),
            ]
            return " ".join(
                [
                    str(attr)
                    for attr in attributes
                    if attr and str(attr).strip() and str(attr) != "NA"
                ]
            )

        # Get embedding for our visitor
        visitor_text = get_visitor_features(visitor)
        if not visitor_text.strip():
            visitor_text = "default visitor profile"

        try:
            # Ensure model is initialized
            self._init_model()
            visitor_embedding = self.model.encode(visitor_text)

            # Calculate similarities for top 20 pre-filtered visitors
            similarities = []
            for vdata in visitors_data:
                v_compare = vdata["v"]
                base_similarity = vdata["base_similarity"]

                compare_text = get_visitor_features(v_compare)
                if not compare_text.strip():
                    continue

                try:
                    compare_embedding = self.model.encode(compare_text)
                    sim = cosine_similarity([visitor_embedding], [compare_embedding])[
                        0
                    ][0]
                    # Combine neural and rule-based similarity
                    combined_sim = (sim * 0.7) + (
                        base_similarity * 0.3 / 4
                    )  # Max base_similarity is 4
                    similarities.append((v_compare["BadgeId"], combined_sim))
                except Exception as e:
                    self.logger.error(
                        f"Error comparing with visitor {v_compare['BadgeId']}: {e}"
                    )
                    continue

            # Sort by similarity (highest first) and get top N
            similarities.sort(key=lambda x: -x[1])
            similar_visitors = [sid for sid, _ in similarities[:num_similar_visitors]]

            # Cache for future use
            self._similar_visitors_cache[visitor_id] = similar_visitors
            return similar_visitors

        except Exception as e:
            self.logger.error(f"Error encoding visitor profile: {e}")
            return []

    def get_similar_visitor_sessions_batch(self, tx, similar_visitor_badge_ids):
        """
        Get sessions attended by similar visitors using a batch query.
        """
        if not similar_visitor_badge_ids:
            return []

        # Single query to get all sessions at once
        query = """
        MATCH (v:Visitor_this_year)-[:Same_Visitor]->(vp)-[:attended_session]->(sp:Sessions_past_year)
        WHERE v.BadgeId IN $similar_visitor_ids AND 
              (vp:Visitor_last_year_bva OR vp:Visitor_last_year_lva)
        RETURN DISTINCT sp.session_id AS session_id, sp.embedding AS embedding
        """

        results = tx.run(query, similar_visitor_ids=similar_visitor_badge_ids).data()

        # Process embeddings
        sessions = []
        for r in results:
            embedding = np.array(json.loads(r["embedding"])) if r["embedding"] else None
            if embedding is not None:
                sessions.append({"session_id": r["session_id"], "embedding": embedding})

        return sessions

    def calculate_session_similarities_parallel(
        self, past_sessions, this_year_sessions, min_score=0.0
    ):
        """
        Calculate similarities between past sessions and this year's sessions in parallel.

        Args:
            past_sessions: List of past session objects with embeddings
            this_year_sessions: Dict of this year's session objects with embeddings
            min_score: Minimum similarity score threshold

        Returns:
            List of recommended sessions with similarity scores
        """
        if not past_sessions or not this_year_sessions:
            return []

        # Function to calculate similarity for a single past session against all this year sessions
        def process_past_session(past_sess):
            recommendations = []
            past_emb = past_sess["embedding"]

            for sid, current_sess in this_year_sessions.items():
                try:
                    current_emb = current_sess["embedding"]
                    sim = cosine_similarity([past_emb], [current_emb])[0][0]

                    if sim >= min_score:
                        recommendations.append(
                            {
                                "session_id": sid,
                                "title": current_sess["title"],
                                "stream": current_sess["stream"],
                                "theatre__name": current_sess["theatre__name"],
                                "date": current_sess["date"],
                                "start_time": current_sess["start_time"],
                                "end_time": current_sess["end_time"],
                                "sponsored_by": current_sess.get("sponsored_by", ""),
                                "sponsored_session": current_sess.get(
                                    "sponsored_session", ""
                                ),
                                "similarity": sim,
                            }
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error calculating similarity for session {sid}: {e}"
                    )

            return recommendations

        # Use parallel processing for faster calculation
        all_recommendations = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(8, len(past_sessions))
        ) as executor:
            future_to_session = {
                executor.submit(process_past_session, ps): ps for ps in past_sessions
            }
            for future in concurrent.futures.as_completed(future_to_session):
                try:
                    recommendations = future.result()
                    all_recommendations.extend(recommendations)
                except Exception as e:
                    self.logger.error(f"Error processing session: {e}")

        # Deduplicate recommendations, keeping the highest similarity score
        session_to_best_rec = {}
        for rec in all_recommendations:
            sid = rec["session_id"]
            if (
                sid not in session_to_best_rec
                or rec["similarity"] > session_to_best_rec[sid]["similarity"]
            ):
                session_to_best_rec[sid] = rec

        # Convert back to list and sort by similarity
        recommendations = list(session_to_best_rec.values())
        recommendations.sort(key=lambda x: -x["similarity"])

        return recommendations

    def recommend_sessions_optimized(
        self,
        badge_id,
        assist_year_before="1",
        min_score=0.0,
        max_recommendations=None,
        num_similar_visitors=3,
    ):
        """
        Optimized version of the recommend_sessions function.

        Args:
            badge_id: Visitor's badge ID
            assist_year_before: "1" if visitor attended last year, "0" otherwise
            min_score: Minimum similarity score for recommendations (0.0-1.0)
            max_recommendations: Maximum number of recommendations to return
            num_similar_visitors: Number of similar visitors to consider

        Returns:
            List of recommended sessions with details
        """
        start_time = time.time()

        # Initialize Neo4j driver and get recommendations
        with GraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        ) as driver:
            with driver.session() as session:
                # Get visitor information
                visitor_info = session.execute_read(
                    self.get_visitor_info, visitor_id=badge_id
                )
                if not visitor_info:
                    self.logger.warning(f"Visitor with BadgeId {badge_id} not found.")
                    return []

                visitor = visitor_info["visitor"]
                assisted = visitor_info["assisted"]

                # Get all this year's sessions in one go (uses caching)
                this_year_sessions = session.execute_read(self.get_this_year_sessions)
                self.logger.info(
                    f"Loaded {len(this_year_sessions)} sessions for this year in {time.time() - start_time:.2f}s"
                )

                past_sessions = []

                if assisted == "1":
                    # Case 1: Visitor attended last year
                    self.logger.info(f"Case 1: Visitor {badge_id} attended last year")
                    case_start = time.time()

                    # Get sessions the visitor attended last year
                    past_sessions = session.execute_read(
                        self.get_past_sessions, visitor_id=badge_id
                    )
                    self.logger.info(
                        f"Found {len(past_sessions)} past sessions in {time.time() - case_start:.2f}s"
                    )

                else:
                    # Case 2: New visitor - find similar visitors
                    self.logger.info(
                        f"Case 2: Finding {num_similar_visitors} similar visitors for {badge_id}"
                    )
                    case_start = time.time()

                    # Find similar visitors in batch
                    similar_visitors = session.execute_read(
                        self.find_similar_visitors_batch,
                        visitor=visitor,
                        num_similar_visitors=num_similar_visitors,
                    )

                    self.logger.info(
                        f"Found {len(similar_visitors)} similar visitors in {time.time() - case_start:.2f}s"
                    )
                    sim_sessions_start = time.time()

                    if similar_visitors:
                        # Get sessions attended by similar visitors in batch
                        past_sessions = session.execute_read(
                            self.get_similar_visitor_sessions_batch,
                            similar_visitor_badge_ids=similar_visitors,
                        )

                        self.logger.info(
                            f"Found {len(past_sessions)} sessions from similar visitors in {time.time() - sim_sessions_start:.2f}s"
                        )

                # Calculate similarities in parallel
                sim_calc_start = time.time()
                recommendations = self.calculate_session_similarities_parallel(
                    past_sessions=past_sessions,
                    this_year_sessions=this_year_sessions,
                    min_score=min_score,
                )
                self.logger.info(
                    f"Calculated similarities in {time.time() - sim_calc_start:.2f}s"
                )

                # Apply maximum recommendations limit
                if max_recommendations and len(recommendations) > max_recommendations:
                    recommendations = recommendations[:max_recommendations]

                self.logger.info(
                    f"Total recommendation time: {time.time() - start_time:.2f}s"
                )
                return recommendations

    def _contains_any(self, text, keywords):
        """Check if text contains any of the keywords (case-insensitive)."""
        if not text or not isinstance(text, str):
            return False

        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)

    def _apply_practice_type_rules(self, visitor, sessions):
        """Apply practice type filtering rules."""
        if not visitor or "what_type_does_your_practice_specialise_in" not in visitor:
            return sessions, []  # No filtering if practice type is missing

        practice_type = visitor.get("what_type_does_your_practice_specialise_in", "")
        if not practice_type or practice_type == "NA":
            return sessions, []

        filtered_sessions = []
        rules_applied = []

        # Check if practice contains equine or mixed
        if self._contains_any(practice_type, ["equine", "mixed"]):
            exclusions = self.rules_config["equine_mixed_exclusions"]
            # Filter out sessions with excluded streams
            filtered_sessions = [
                session
                for session in sessions
                if not session.get("stream")
                or not self._contains_any(session["stream"], exclusions)
            ]
            rules_applied.append(
                f"practice_type: mixed/equine - excluded {', '.join(exclusions)}"
            )
            self.logger.info(
                f"Applied equine/mixed rule: filtered from {len(sessions)} to {len(filtered_sessions)} sessions"
            )

        # Check if practice contains small animal
        elif self._contains_any(practice_type, ["small animal"]):
            exclusions = self.rules_config["small_animal_exclusions"]
            # Filter out sessions with excluded streams
            filtered_sessions = [
                session
                for session in sessions
                if not session.get("stream")
                or not self._contains_any(session["stream"], exclusions)
            ]
            rules_applied.append(
                f"practice_type: small animal - excluded {', '.join(exclusions)}"
            )
            self.logger.info(
                f"Applied small animal rule: filtered from {len(sessions)} to {len(filtered_sessions)} sessions"
            )

        else:
            # No specific practice type rule applies
            filtered_sessions = sessions

        return filtered_sessions, rules_applied

    def _apply_role_rules(self, visitor, sessions):
        """Apply job role filtering rules."""
        if not visitor or "job_role" not in visitor:
            return sessions, []  # No filtering if job role is missing

        job_role = visitor.get("job_role", "")
        if not job_role or job_role == "NA":
            return sessions, []

        filtered_sessions = []
        rules_applied = []

        # Rule for VET_ROLES
        if job_role in self.vet_roles:
            exclusions = self.rules_config["vet_exclusions"]
            # Filter out sessions with excluded streams
            filtered_sessions = [
                session
                for session in sessions
                if not session.get("stream")
                or not self._contains_any(session["stream"], exclusions)
            ]
            rules_applied.append(f"role: vet - excluded {', '.join(exclusions)}")
            self.logger.info(
                f"Applied vet role rule: filtered from {len(sessions)} to {len(filtered_sessions)} sessions"
            )

        # Rule for NURSE_ROLES
        elif job_role in self.nurse_roles:
            allowed_streams = self.rules_config["nurse_streams"]
            # Only keep sessions with allowed streams
            filtered_sessions = [
                session
                for session in sessions
                if session.get("stream")
                and self._contains_any(session["stream"], allowed_streams)
            ]
            rules_applied.append(
                f"role: nurse - limited to {', '.join(allowed_streams)}"
            )
            self.logger.info(
                f"Applied nurse role rule: filtered from {len(sessions)} to {len(filtered_sessions)} sessions"
            )

        else:
            # No specific role rule applies
            filtered_sessions = sessions

        return filtered_sessions, rules_applied

    def filter_sessions(self, visitor, sessions):
        """
        Filter sessions based on visitor profile and business rules.

        Args:
            visitor: Dictionary containing visitor profile
            sessions: List of session dictionaries to filter

        Returns:
            Tuple of (filtered_sessions, rules_applied) where rules_applied is a list of rule descriptions
        """
        if not sessions:
            return [], []

        filtered_sessions = sessions
        rule_priority = self.rules_config["rule_priority"]
        all_rules_applied = []

        # Apply rules in priority order
        for rule_type in rule_priority:
            if rule_type == "practice_type":
                filtered_sessions, rules_applied = self._apply_practice_type_rules(
                    visitor, filtered_sessions
                )
                all_rules_applied.extend(rules_applied)
            elif rule_type == "role":
                filtered_sessions, rules_applied = self._apply_role_rules(
                    visitor, filtered_sessions
                )
                all_rules_applied.extend(rules_applied)
            else:
                self.logger.warning(f"Unknown rule type: {rule_type}")

        # Sort by similarity score (highest first)
        filtered_sessions.sort(
            key=lambda x: float(x.get("similarity", 0)), reverse=True
        )

        return filtered_sessions, all_rules_applied

    def filter_with_langchain(self, visitor, recommendations):
        """
        Filter recommendations using LangChain and LLM.

        Args:
            visitor: Visitor profile dictionary
            recommendations: List of recommended sessions

        Returns:
            Filtered list of recommendations and a list of processing notes
        """
        if not has_langchain:
            self.logger.warning("LangChain not available. Cannot filter using LLM.")
            return recommendations, ["LangChain filtering not available"]

        try:
            # Load .env file from config
            env_file = self.config.get("env_file", ".env")
            config_values = dotenv_values(env_file)

            # Initialize Azure OpenAI client
            llm = AzureChatOpenAI(
                azure_endpoint=config_values.get("AZURE_ENDPOINT", ""),
                azure_deployment=config_values.get("AZURE_DEPLOYMENT", ""),
                api_key=config_values.get("AZURE_API_KEY", ""),
                api_version=config_values.get("AZURE_API_VERSION", ""),
                temperature=0.5,
                top_p=0.9,
            )

            # Get keys from visitor and recommendations
            list_keys_vis = list(visitor.keys())
            if recommendations:
                list_keys = list(recommendations[0].keys())
            else:
                list_keys = [
                    "session_id",
                    "title",
                    "stream",
                    "theatre__name",
                    "date",
                    "start_time",
                    "end_time",
                    "sponsored_by",
                    "sponsored_session",
                    "similarity",
                ]

            # Generate the prompt
            system_prompt = f"""
            you are an assistant specialized in filter sessions of an Event based in bussiness rules and profiles of users.\n
            - you will receive a profile of a visitor with the following keys: {list_keys_vis}\n
              the attribute what_type_does_your_practice_specialise_in can be a list of specializations separated by ";"
            - you will receive a list of session with the following keys: {list_keys}\n
              stream in session can be a list of tpocis separated by ";". When you evaluate the rule you need to consider all of them \n
            - each session you return must have the same format.
            - different Job_Roles Groups:\n
            VET_ROLES = [
            "Vet/Vet Surgeon",
            "Assistant Vet",
            "Vet/Owner",
            "Clinical or other Director",
            "Locum Vet", 
            "Academic",
            ]\n
            
            NURSE_ROLES = ["Head Nurse/Senior Nurse", "Vet Nurse", "Locum RVN"]\n
            
            BUSINESS = ["Practice Manager", "Practice Partner/Owner"]\n
            # Other roles can attend any session
            OTHER_ROLES = ["Student", "Receptionist", "Other (please specify)"]\n
            - only return the sessions in json format
            """

            prompt = PromptTemplate(
                input_variables=["sessions", "rules", "profile"],
                template=system_prompt
                + """For the Visitor with profile {profile}\n, based on the attributes of these session: {sessions} and implementing the following rules {rules}.\n Filter them and just return from the list those that meet the requirements in the rules""",
            )

            # Create chain
            chain = prompt | llm

            # Convert recommendations to text
            text_rec = json.dumps(recommendations)

            # Invoke the chain
            self.logger.info("Invoking LangChain for session filtering")
            ai_msg = chain.invoke(
                {"sessions": text_rec, "profile": visitor, "rules": self.business_rules}
            )

            # Parse the response to extract the filtered recommendations
            response_text = ai_msg.content

            # Try to extract JSON from the response
            import re

            json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without markdown
                json_match = re.search(r"\[\s*{.*}\s*\]", response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response_text

            try:
                filtered_recommendations = json.loads(json_str)
                self.logger.info(
                    f"Successfully filtered recommendations using LangChain: {len(filtered_recommendations)} sessions"
                )
                return filtered_recommendations, ["Filtered using LLM-based rules"]
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing LangChain response: {e}")
                return recommendations, ["Error in LLM filtering"]

        except Exception as e:
            self.logger.error(f"Error using LangChain for filtering: {e}")
            return recommendations, ["Error in LLM filtering: " + str(e)]

    def get_recommendations_and_filter(
        self,
        badge_id,
        min_score=0.5,
        max_recommendations=30,
        visitor_data=None,
        use_langchain=False,
    ):
        """
        Get session recommendations and filter them based on business rules.

        Args:
            badge_id: Visitor's badge ID
            min_score: Minimum similarity score
            max_recommendations: Maximum number of recommendations to return
            visitor_data: Optional visitor data dictionary
            use_langchain: Whether to filter using LangChain

        Returns:
            Dictionary with visitor profile, raw recommendations, filtered recommendations,
            and metadata about the filtering process
        """
        start_time = time.time()
        processing_steps = []

        # Get visitor profile from Neo4j if not provided
        visitor = visitor_data
        if visitor is None:
            try:
                with GraphDatabase.driver(
                    self.uri, auth=(self.username, self.password)
                ) as driver:
                    with driver.session() as session:
                        result = session.run(
                            "MATCH (v:Visitor_this_year {BadgeId: $badge_id}) RETURN v",
                            badge_id=badge_id,
                        ).single()

                        if result:
                            visitor = dict(result["v"])
                            processing_steps.append(
                                f"Found visitor {badge_id} in Neo4j"
                            )
                        else:
                            processing_steps.append(
                                f"Visitor {badge_id} not found in Neo4j"
                            )
            except Exception as e:
                self.logger.error(f"Error connecting to Neo4j: {e}")
                processing_steps.append(f"Neo4j connection error: {e}")

        # If visitor still not found, return empty result
        if visitor is None:
            return {
                "visitor": None,
                "raw_recommendations": [],
                "filtered_recommendations": [],
                "metadata": {
                    "error": "Visitor not found",
                    "processing_time": time.time() - start_time,
                    "processing_steps": processing_steps,
                },
            }

        # Get recommendations
        try:
            # Get raw recommendations
            recommendations = self.recommend_sessions_optimized(
                badge_id=badge_id,
                assist_year_before=str(visitor.get("assist_year_before", "1")),
                min_score=min_score,
                max_recommendations=max_recommendations,
            )

            processing_steps.append(
                f"Retrieved {len(recommendations)} raw recommendations"
            )

            # Choose filtering approach
            if use_langchain and has_langchain:
                # Use LangChain filtering
                processing_steps.append("Using LangChain for filtering")
                filtered_recommendations, filter_notes = self.filter_with_langchain(
                    visitor=visitor, recommendations=recommendations
                )
                processing_steps.extend(filter_notes)
            else:
                # Use rule-based filtering
                processing_steps.append("Using rule-based filtering")
                filtered_recommendations, filter_notes = self.filter_sessions(
                    visitor, recommendations
                )
                processing_steps.extend(filter_notes)

            processing_steps.append(
                f"Filtered to {len(filtered_recommendations)} recommendations"
            )

            # Create result dictionary
            result = {
                "visitor": visitor,
                "raw_recommendations": recommendations,
                "filtered_recommendations": filtered_recommendations,
                "metadata": {
                    "badge_id": badge_id,
                    "num_raw_recommendations": len(recommendations),
                    "num_filtered_recommendations": len(filtered_recommendations),
                    "processing_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat(),
                    "processing_steps": processing_steps,
                },
            }

            return result

        except Exception as e:
            self.logger.error(f"Error getting recommendations: {str(e)}")
            return {
                "visitor": visitor,
                "raw_recommendations": [],
                "filtered_recommendations": [],
                "metadata": {
                    "error": str(e),
                    "processing_time": time.time() - start_time,
                    "processing_steps": processing_steps,
                },
            }
