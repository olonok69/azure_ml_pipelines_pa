# Implementation Guide: Azure ML Pipeline for Personal Agendas

## Overview
This guide provides step-by-step instructions to implement the Azure ML pipeline for the Personal Agendas system, starting with Step 1 (Data Preparation).

## Architecture
```
Current PA Pipeline (10 steps) → Azure ML Pipeline (4 steps)
├── Step 1: Data Preparation
│   ├── Registration Processing
│   ├── Scan Processing  
│   └── Session Processing
├── Step 2: Neo4j Preparation
│   ├── Visitor Processing
│   ├── Session Processing
│   ├── Job Stream Processing
│   ├── Specialization Stream Processing
│   └── Visitor Relationship Processing
├── Step 3: Session Embeddings
└── Step 4: Recommendations
```

## Prerequisites
1. Azure ML Workspace configured
2. Azure Blob Storage with input data
3. Neo4j database connection details
4. Python 3.10+ environment

## Directory Structure
```
project_root/
├── PA/                           # Existing PA code (unchanged)
│   ├── main.py
│   ├── pipeline.py
│   ├── registration_processor.py
│   ├── scan_processor.py
│   ├── session_processor.py
│   ├── config/
│   │   ├── config_vet.yaml
│   │   ├── config_vet_lva.yaml   # current target show
│   │   └── config_ecomm.yaml
│   └── utils/
├── azureml_pipeline/            # New Azure ML pipeline code
│   ├── azureml_step1_data_prep.py
│   ├── azureml_step2_neo4j_prep.py
│   ├── azureml_step3_session_embedding.py
│   ├── azureml_step4_recommendations.py
│   └── pipeline_config.yaml
├── env/
│   └── conda.yaml
└── notebooks/
    └── submit_pipeline_complete.ipynb
```

## Implementation Steps

### Step 1: Environment Setup

1. **Create the directory structure**:
```bash
mkdir -p azureml_pipeline
mkdir -p env
mkdir -p notebooks
```

2. **Copy the PA code** (without modifications):
```bash
cp -r /path/to/PA ./PA
```

3. **Place the new Azure ML files**:
   - Save `azureml_step1_data_prep.py` to `azureml_pipeline/`
   - Save `pipeline_config.yaml` to `azureml_pipeline/`
   - Save `conda.yaml` to `env/`

### Step 2: Configure Azure ML Environment

1. **Create the Azure ML environment**:
```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# Initialize ML Client
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential,
    subscription_id="<your-subscription-id>",
    resource_group="<your-resource-group>",
    workspace="<your-workspace>"
)

# Create or update the shared PA environment (CPU only)
env = Environment(
    name="pa-env",
    description="CPU environment for Personal Agendas pipeline",
    conda_file="./env/conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20250601.v1"
)
ml_client.environments.create_or_update(env)
```

### Step 3: Create the Pipeline Submission Script

Use `notebooks/submit_pipeline_complete.ipynb` (run from the `notebooks/` folder so it can reference the repo root). The notebook already wires all four Azure ML command components to the real scripts:

- Step 1 → `azureml_step1_data_prep.py`
- Step 2 → `azureml_step2_neo4j_prep.py`
- Step 3 → `azureml_step3_session_embedding.py`
- Step 4 → `azureml_step4_recommendations.py`

Key settings for the current run:

1. **Config selection** – set `pipeline_config_type="vet_lva"` so that every step receives `PA/config/config_vet_lva.yaml`.
2. **Input data** – point `pipeline_input_data` at the `landing_pa` datastore path that contains the vetted landing files:

```python
input_data_uri = (
    f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/"
    f"workspaces/{workspace_name}/datastores/landing_pa/paths/landing/azureml/"
)

pipeline_job = personal_agendas_complete_pipeline(
    pipeline_input_data=Input(type=AssetTypes.URI_FOLDER, path=input_data_uri),
    pipeline_config_type="vet_lva",
    pipeline_incremental="false"
)
```

3. **Compute / environment** – all four steps target the shared `cpu-cluster` and use the `pa-env` Azure ML environment, which matches the Standard_E4ds_v4 compute instance available in the workspace.
4. **Secrets** – the notebook passes `KEYVAULT_NAME`, Neo4j credentials, and service-principal variables through `environment_variables` so each step can hydrate `PA/keys/.env` on the fly. Ensure those secrets exist in Key Vault before submission.

Run the bottom cells of the notebook to submit, monitor, and log the pipeline run in Azure ML Studio.

### Step 4: Data Transfer Between Steps

Azure ML automatically persists every step output in the workspace datastore. The pipeline relies on those managed folders to keep the run reproducible:

1. Step 1 writes three hand-off folders (`registration_output`, `scan_output`, `session_output`) plus a `metadata_output` manifest. All downstream steps stay dependent on these folders so we can rehydrate Neo4j from the same artifacts.
2. Step 2 consumes the three folders, recreates the PA `data/<event>/output` structure, loads Neo4j, and publishes a single `metadata_output` folder that records processor statistics. Step 3 is configured to depend on that metadata output to enforce ordering even though embeddings read directly from Neo4j.
3. Step 3 emits another `metadata_output` folder so that Step 4 will not start until embeddings have completed for the target show.

The actual folder names map 1:1 to the component outputs inside `submit_pipeline_complete.ipynb` and to the `steps` definitions in `azureml_pipeline/pipeline_config.yaml`.

## Step Contracts (Inputs / Outputs)

| Step | Inputs | Outputs | Notes |
| --- | --- | --- | --- |
| Step 1 – Data Preparation | `pipeline_input_data` (landing_pa folder mounted read-only), `PA/config/config_<event>.yaml`, `incremental` flag | `registration_output`, `scan_output`, `session_output`, `metadata_output` (each an Azure ML folder) | The script copies curated CSV/JSON artifacts into the output folders with canonical names that remain stable across shows. `metadata_output` contains processor logs and run stats. |
| Step 2 – Neo4j Preparation | Step 1 registration/scan/session outputs, same config file, `incremental` flag | `metadata_output` | Copies Step 1 artifacts back into `data/<event>/output`, loads Neo4j, and records processor statistics. No graph export is produced yet, so replays depend on the Step 1 folders. |
| Step 3 – Session Embedding | Config file, `incremental` flag, implicit dependency on Step 2 metadata | `metadata_output` | Reads session/state directly from Neo4j. The output folder contains embedding summaries plus a completion marker so Step 4 only runs after embeddings finish. |
| Step 4 – Recommendations | Config file, `incremental` flag, implicit dependency on Step 3 metadata | `metadata_output` (includes copied JSON/CSV recommendation files) | The processor drops recommendation exports under `data/output/recommendations` and the step copies the latest files into the Azure ML output so they can be downloaded from the pipeline job. |

Use this table as the contract when validating new shows—if a step is re-run manually, ensure the expected input folders exist (from either the previous step or from an earlier pipeline run) before starting Azure ML.

### Step 5: Configuration Management

The pipeline configuration is managed through:

1. **YAML config files** (`config_vet.yaml`, `config_vet_lva.yaml`, `config_ecomm.yaml`) - unchanged from PA
2. **Pipeline parameters** - passed via Azure ML pipeline
3. **Environment variables** - for Azure credentials

To switch between VET and ECOMM configurations:
```python
# For Vet LVA (current run)
pipeline_job = personal_agendas_complete_pipeline(
   pipeline_input_data=<landing_pa path>,
   pipeline_config_type="vet_lva"
)

# For other events adjust only the config selector
pipeline_job = personal_agendas_complete_pipeline(
   pipeline_input_data=<landing_pa path>,
   pipeline_config_type="vet"  # or "ecomm"
)
```

### Step 6: Incremental Processing

Enable incremental processing by passing the parameter:
```python
step1 = data_preparation_step(
    input_uri=input_data_uri,
    config_file=config_file,
    incremental=True  # Enable incremental processing
)
```

### Step 7: Monitoring and Debugging

1. **View logs** in Azure ML Studio:
   - Navigate to the pipeline run
   - Click on each step to view logs
   - Check "Outputs + logs" tab

2. **MLflow tracking**:
   - Metrics are automatically logged
   - View in Azure ML Studio's Metrics tab

3. **Debug locally** before deploying:
```bash
# Test the step locally
python azureml_pipeline/azureml_step1_data_prep.py \
    --config PA/config/config_vet.yaml \
    --input_uri "path/to/local/data"
```

## Key Design Decisions

### 1. No Code Changes to PA
- The Azure ML scripts import and use PA processors directly
- Configuration files remain unchanged
- Maintains backward compatibility

### 2. Generic Pipeline Design
- Configuration-driven (YAML files)
- Supports multiple event types (VET, ECOMM)
- Incremental processing capability

### 3. Data Transfer Strategy
- Uses Azure ML Output objects for inter-step communication
- Automatic data versioning and lineage tracking
- Efficient blob storage transfer

### 4. Error Handling
- Comprehensive logging at each step
- Retry logic for transient failures
- Metadata tracking for debugging

## Testing Strategy

1. **Unit Testing**:
   - Test each processor independently
   - Verify output file generation

2. **Integration Testing**:
   - Test complete pipeline with sample data
   - Verify data flow between steps

3. **Performance Testing**:
   - Monitor processing times
   - Optimize compute resources

## Next Steps

1. **Implement Step 2** (Neo4j Preparation):
   - Similar structure to Step 1
   - Uses outputs from Step 1

2. **Implement Step 3** (Session Embeddings):
   - Runs today on CPU `pa-env` / `cpu-cluster`
   - If a GPU SKU becomes available later, only the component compute target needs to change

3. **Implement Step 4** (Recommendations):
   - Final output generation
   - Metrics and evaluation

## Troubleshooting

### Common Issues and Solutions

1. **Import errors**:
   - Ensure PA directory is in Python path
   - Check conda environment dependencies

2. **Configuration not found**:
   - Verify config file paths
   - Check Azure ML datastore mounts

3. **Memory issues**:
   - Increase compute cluster VM size
   - Enable batch processing

4. **Neo4j connection failures**:
   - Store credentials in Azure Key Vault
   - Use managed identity for authentication

## Support and Resources

- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/)