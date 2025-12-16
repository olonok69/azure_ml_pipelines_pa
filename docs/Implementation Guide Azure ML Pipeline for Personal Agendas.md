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
│   │   └── config_ecomm.yaml
│   └── utils/
├── azureml_pipeline/            # New Azure ML pipeline code
│   ├── azureml_step1_data_prep.py
│   ├── azureml_step2_neo4j_prep.py
│   ├── azureml_step3_embeddings.py
│   ├── azureml_step4_recommendations.py
│   └── pipeline_config.yaml
├── env/
│   └── conda.yaml
└── notebooks/
    └── submit_pipeline.ipynb
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

# Create environment
env = Environment(
    name="personal_agendas_env",
    description="Environment for Personal Agendas pipeline",
    conda_file="./env/conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)
ml_client.environments.create_or_update(env)
```

### Step 3: Create the Pipeline Submission Script

Create a Jupyter notebook or Python script to submit the pipeline:

```python
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential
import os

# Initialize ML Client
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential,
    subscription_id=os.getenv("SUBSCRIPTION_ID"),
    resource_group=os.getenv("RESOURCE_GROUP"),
    workspace=os.getenv("AZUREML_WORKSPACE_NAME")
)

# Define Step 1: Data Preparation
@command(
    name="data_preparation",
    display_name="Data Preparation (Registration, Scan, Session)",
    environment="personal_agendas_env:latest",
    compute="cpu-cluster",
    code="./azureml_pipeline",
    is_deterministic=False
)
def data_preparation_step(
    input_uri: Input(type=AssetTypes.URI_FOLDER),
    config_file: Input(type=AssetTypes.URI_FILE),
    incremental: bool = False
) -> dict:
    return {
        "registration_output": Output(type=AssetTypes.URI_FOLDER),
        "scan_output": Output(type=AssetTypes.URI_FOLDER),
        "session_output": Output(type=AssetTypes.URI_FOLDER),
        "metadata_output": Output(type=AssetTypes.URI_FOLDER)
    }

# Define the pipeline
@pipeline(
    name="personal_agendas_pipeline",
    description="Personal Agendas data processing pipeline"
)
def personal_agendas_pipeline(
    input_data_uri: str,
    config_type: str = "vet"  # or "ecomm"
):
    # Step 1: Data Preparation
    step1 = data_preparation_step(
        input_uri=input_data_uri,
        config_file=f"./PA/config/config_{config_type}.yaml",
        incremental=False
    )
    
    # Additional steps would be added here
    # step2 = neo4j_preparation_step(...)
    # step3 = embeddings_step(...)
    # step4 = recommendations_step(...)
    
    return {
        "registration_data": step1.outputs.registration_output,
        "scan_data": step1.outputs.scan_output,
        "session_data": step1.outputs.session_output
    }

# Submit the pipeline
pipeline_job = personal_agendas_pipeline(
    input_data_uri="azureml://datastores/landing/paths/weekly_refresh_data",
    config_type="vet"
)

# Submit to Azure ML
submitted_job = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="personal_agendas_experiment"
)

print(f"Pipeline submitted: {submitted_job.name}")
print(f"Monitor at: {submitted_job.studio_url}")
```

### Step 4: Data Transfer Between Steps

The pipeline uses Azure ML's Output objects to transfer data between steps:

1. **Step 1 outputs** are automatically saved to Azure Blob Storage
2. **Step 2 inputs** reference Step 1's outputs using `source` parameter
3. Azure ML handles the data transfer automatically

Example for Step 2 input configuration:
```python
step2_inputs = {
    "registration_data": step1.outputs.registration_output,
    "scan_data": step1.outputs.scan_output,
    "session_data": step1.outputs.session_output
}
```

### Step 5: Configuration Management

The pipeline configuration is managed through:

1. **YAML config files** (config_vet.yaml, config_ecomm.yaml) - unchanged from PA
2. **Pipeline parameters** - passed via Azure ML pipeline
3. **Environment variables** - for Azure credentials

To switch between VET and ECOMM configurations:
```python
# For VET events
pipeline_job = personal_agendas_pipeline(
    input_data_uri="...",
    config_type="vet"
)

# For ECOMM events  
pipeline_job = personal_agendas_pipeline(
    input_data_uri="...",
    config_type="ecomm"
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
   - Requires GPU compute
   - Separate environment with ML libraries

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