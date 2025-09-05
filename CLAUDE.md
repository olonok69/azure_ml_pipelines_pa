# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Generic Event Recommendation Pipeline (Personal Agendas) - a configurable data processing pipeline for event management that:
- Processes event registration and attendance data
- Generates personalized session recommendations using ML/AI
- Builds comprehensive knowledge graphs in Neo4j
- Supports multiple event types through YAML configuration files

The codebase was refactored from a veterinary-specific system to a generic, configuration-driven architecture.

## High-Level Architecture

### Core Pipeline Flow
1. **Data Processing Phase (Steps 1-3)**: Registration, scan, and session data processing
2. **Neo4j Integration Phase (Steps 4-8)**: Create nodes and relationships in knowledge graph
3. **ML/AI Processing Phase (Steps 9-10)**: Generate embeddings and recommendations

### Key Components
- **PA/** - Core Python package (renamed from 'app')
  - `main.py` - Main orchestrator with MLflow integration
  - `pipeline.py` - Pipeline coordinator
  - Data processors: `registration_processor.py`, `scan_processor.py`, `session_processor.py`
  - Neo4j processors: `neo4j_visitor_processor.py`, `neo4j_session_processor.py`, etc.
  - ML processors: `session_embedding_processor.py`, `session_recommendation_processor.py`
- **config/** - Event-specific YAML configuration files
- **azureml_pipeline/** - Azure ML pipeline step definitions

### Configuration-Driven Design
The pipeline behavior is controlled by YAML configuration files (e.g., `config_vet.yaml`, `config_ecomm.yaml`) that specify:
- Event details and show identifiers
- Field mappings for different event types
- Pipeline step activation/deactivation
- Neo4j connection details
- ML model parameters

## Common Commands

### Running the Pipeline

```bash
# Run complete pipeline with veterinary configuration
python PA/main.py --config PA/config/config_vet.yaml

# Run with e-commerce configuration
python PA/main.py --config PA/config/config_ecomm.yaml

# Run specific steps only
python PA/main.py --config PA/config/config_vet.yaml --only-steps 1,2,3

# Skip Neo4j upload
python PA/main.py --config PA/config/config_vet.yaml --skip-neo4j

# Process only new visitors without existing recommendations
python PA/main.py --config PA/config/config_vet.yaml --create-only-new

# Recreate all Neo4j nodes
python PA/main.py --config PA/config/config_vet.yaml --recreate-all
```

### Standalone Components

```bash
# Generate session embeddings only
python PA/run_embedding.py --config PA/config/config_vet.yaml

# Generate recommendations only
python PA/run_recommendations.py --config PA/config/config_vet.yaml --min-score 0.3 --max-recommendations 10
```

### Azure ML Pipeline Steps

```bash
# Step 1: Data preparation
python azureml_pipeline/azureml_step1_data_prep.py --config config/config_vet.yaml

# Step 2: Neo4j preparation
python azureml_pipeline/azureml_step2_neo4j_prep.py --config config/config_vet.yaml

# Step 3: Session embeddings
python azureml_pipeline/azureml_step3_session_embedding.py --config config/config_vet.yaml

# Step 4: Recommendations
python azureml_pipeline/azureml_step4_recommendations.py --config config/config_vet.yaml
```

### Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment variables (create keys/.env file)
# Required: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
# Optional: OPENAI_API_KEY or AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT
```

## Key Dependencies and Requirements

- Python 3.8+
- Neo4j Database 4.4+
- Key Python packages: pandas, neo4j, sentence-transformers, openai, langchain, pyyaml
- Azure Key Vault integration (see `utils/keyvault_utils.py`)
- MLflow for experiment tracking (optional)

## Important Notes

1. **Configuration Files**: The pipeline behavior is entirely driven by YAML config files in `PA/config/`. Always check and update the correct config file for your event type.

2. **Neo4j Schema**: The pipeline creates specific node types (`Visitor_this_year`, `Visitor_last_year_bva`, `Sessions_this_year`, etc.) and relationships (`HAS_STREAM`, `attended_session`, `Same_Visitor`, etc.) as defined in the config.

3. **Data Flow**: Raw data from `data/[event_name]/` → Processed CSVs → Neo4j nodes → ML embeddings → Recommendations

4. **Event-Specific Logic**: While the processors are generic, some vet-specific functions still exist in `utils/vet_specific_functions.py`. The pipeline detects if `main_event_name == "bva"` to trigger these.

5. **Azure Integration**: The codebase includes Azure ML pipeline steps and Azure Key Vault integration for secure credential management.

6. **MLflow Tracking**: The pipeline includes comprehensive MLflow integration for tracking experiments, parameters, and metrics. Use `--skip-mlflow` to disable.

7. **Environment Variables**: Sensitive credentials should be stored in `keys/.env` file (not committed to git).

8. **Logging**: Comprehensive logging is available in `PA/data_processing.log` and through MLflow UI.