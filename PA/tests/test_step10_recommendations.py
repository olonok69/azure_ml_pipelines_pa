import logging
from pathlib import Path

from utils.config_utils import load_config
from session_recommendation_processor import SessionRecommendationProcessor


def run_step10_smoke_test():
    """Run a small smoke test of step 10 (recommendations).

    - Loads the LVA post-analysis config
    - Instantiates SessionRecommendationProcessor (this validates Neo4j env)
    - Runs .process(create_only_new=True) so it only touches visitors
      without existing recommendations.
    """
    logging.basicConfig(level=logging.INFO)

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "config" / "config_vet_lva.yaml"

    print(f"Using config: {config_path}")

    config = load_config(str(config_path))

    processor = SessionRecommendationProcessor(config)

    print("Starting step 10 smoke test (create_only_new=True)...")
    processor.process(create_only_new=True)

    print("Step 10 smoke test finished.")
    print("Statistics (high level):")
    for key in [
        "visitors_processed",
        "visitors_with_recommendations",
        "visitors_without_recommendations",
        "total_recommendations_generated",
        "unique_recommended_sessions",
        "errors",
    ]:
        if key in processor.statistics:
            print(f"  {key}: {processor.statistics[key]}")


if __name__ == "__main__":
    run_step10_smoke_test()
