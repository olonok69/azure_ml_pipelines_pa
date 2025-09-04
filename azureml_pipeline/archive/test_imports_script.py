"""
Simple test to verify setup and submit pipeline with fixed imports
"""

from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential, ClientSecretCredential
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

print("="*60)
print("AZURE ML PIPELINE SUBMISSION WITH FIXES")
print("="*60)

# Get environment variables
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("AZUREML_WORKSPACE_NAME")

print(f"Subscription: {subscription_id}")
print(f"Resource Group: {resource_group}")
print(f"Workspace: {workspace_name}")

# Verify all values are present
if not all([subscription_id, resource_group, workspace_name]):
    print("\n❌ ERROR: Missing environment variables!")
    print("Please check your .env file contains:")
    print("  SUBSCRIPTION_ID=...")
    print("  RESOURCE_GROUP=...")
    print("  AZUREML_WORKSPACE_NAME=...")
    exit(1)

# Create credential - try both methods
try:
    if os.getenv("AZURE_CLIENT_ID") and os.getenv("AZURE_CLIENT_SECRET") and os.getenv("AZURE_TENANT_ID"):
        print("\nUsing Service Principal authentication")
        credential = ClientSecretCredential(
            tenant_id=os.getenv("AZURE_TENANT_ID"),
            client_id=os.getenv("AZURE_CLIENT_ID"),
            client_secret=os.getenv("AZURE_CLIENT_SECRET")
        )
    else:
        print("\nUsing DefaultAzureCredential")
        credential = DefaultAzureCredential()
except Exception as e:
    print(f"Error creating credential: {e}")
    exit(1)

# Initialize ML Client with explicit values
try:
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    print(f"✓ Connected to workspace: {ml_client.workspace_name}")
except Exception as e:
    print(f"❌ Error connecting to workspace: {e}")
    exit(1)

# Determine project root
current_dir = Path.cwd()
if current_dir.name == "azureml_pipeline":
    project_root = current_dir.parent
elif current_dir.name == "notebooks":
    project_root = current_dir.parent
else:
    project_root = current_dir

print(f"Project root: {project_root}")

# Check if directories exist
pa_dir = project_root / "PA"
pipeline_dir = project_root / "azureml_pipeline"

if not pa_dir.exists():
    print(f"❌ PA directory not found at {pa_dir}")
    print("Please ensure you're running from the correct directory")
    exit(1)
else:
    print(f"✓ PA directory found")

if not pipeline_dir.exists():
    print(f"❌ azureml_pipeline directory not found at {pipeline_dir}")
    exit(1)
else:
    print(f"✓ azureml_pipeline directory found")

# First, let's fix the azureml_step1_data_prep.py file
step1_file = pipeline_dir / "azureml_step1_data_prep.py"
if step1_file.exists():
    print(f"\n📝 Updating import paths in {step1_file.name}...")
    
    # Read the current content
    with open(step1_file, 'r') as f:
        content = f.read()
    
    # Check if we need to fix imports
    if "from registration_processor import" in content and "sys.path.insert" not in content:
        print("  Fixing imports...")
        
        # Find the import section
        import_fix = '''import os
import sys

# Fix import paths for Azure ML environment
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
pa_dir = os.path.join(parent_dir, 'PA')

# Add paths
sys.path.insert(0, parent_dir)
sys.path.insert(0, pa_dir)

# Now imports will work - try both methods
try:
    from registration_processor import RegistrationProcessor
    from scan_processor import ScanProcessor
    from session_processor import SessionProcessor
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging
except ImportError:
    # Fallback to PA prefix
    from PA.registration_processor import RegistrationProcessor
    from PA.scan_processor import ScanProcessor
    from PA.session_processor import SessionProcessor
    from PA.utils.config_utils import load_config
    from PA.utils.logging_utils import setup_logging
'''
        
        # Replace the import section
        content = content.replace(
            "from registration_processor import RegistrationProcessor",
            import_fix
        )
        
        # Remove duplicate imports
        lines_to_remove = [
            "from scan_processor import ScanProcessor",
            "from session_processor import SessionProcessor",
            "from utils.config_utils import load_config",
            "from utils.logging_utils import setup_logging"
        ]
        
        for line in lines_to_remove:
            content = content.replace(line, "")
        
        # Save the fixed file
        with open(step1_file, 'w') as f:
            f.write(content)
        
        print("  ✓ Imports fixed!")
    else:
        print("  ✓ Imports already fixed or different format")

print("\n" + "="*60)
print("SUBMITTING PIPELINE")
print("="*60)

# Get or create environment
try:
    env_name = "personal_agendas_env"
    envs = list(ml_client.environments.list(name=env_name))
    if envs:
        env_version = envs[0].version
        print(f"Using existing environment: {env_name}:{env_version}")
    else:
        print("Creating new environment...")
        env = Environment(
            name=env_name,
            description="Environment for Personal Agendas pipeline",
            conda_file=str(project_root / "env" / "conda.yaml"),
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
        )
        env = ml_client.environments.create_or_update(env)
        env_version = env.version
        print(f"Created environment: {env_name}:{env_version}")
except Exception as e:
    print(f"Environment setup issue: {e}")
    print("Using default environment instead")
    env_name = "AzureML-sklearn-1.0-ubuntu20.04-py38-cpu"
    env_version = "1"

# Create the data preparation component
data_preparation_component = command(
    name="data_preparation",
    display_name="Data Preparation - Fixed Imports",
    description="Process registration, scan, and session data",
    inputs={
        "input_uri": Input(type="uri_folder"),
        "config_type": Input(type="string", default="ecomm"),
        "incremental": Input(type="boolean", default=False)
    },
    outputs={
        "registration_output": Output(type="uri_folder"),
        "scan_output": Output(type="uri_folder"),
        "session_output": Output(type="uri_folder"),
        "metadata_output": Output(type="uri_folder")
    },
    code=str(project_root),  # Upload entire project
    command="""python azureml_pipeline/azureml_step1_data_prep.py \
        --config PA/config/config_${{inputs.config_type}}.yaml \
        --input_uri ${{inputs.input_uri}} \
        --incremental ${{inputs.incremental}} \
        --output_registration ${{outputs.registration_output}} \
        --output_scan ${{outputs.scan_output}} \
        --output_session ${{outputs.session_output}} \
        --output_metadata ${{outputs.metadata_output}}
    """,
    environment=f"{env_name}:{env_version}",
    compute="cpu-cluster",
    is_deterministic=False
)

# Define the pipeline
@pipeline(
    compute="cpu-cluster",
    description="Personal Agendas data processing pipeline",
)
def personal_agendas_pipeline(
    pipeline_input_data: Input,
    pipeline_config_type: str = "ecomm",
    pipeline_incremental: bool = False
):
    step1 = data_preparation_component(
        input_uri=pipeline_input_data,
        config_type=pipeline_config_type,
        incremental=pipeline_incremental
    )
    
    return {
        "registration_data": step1.outputs.registration_output,
        "scan_data": step1.outputs.scan_output,
        "session_data": step1.outputs.session_output,
        "metadata": step1.outputs.metadata_output
    }

# Create pipeline instance
input_data_uri = f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}/datastores/azureml_landing/paths/landing/azureml/ecomm/"

pipeline_job = personal_agendas_pipeline(
    pipeline_input_data=Input(
        type=AssetTypes.URI_FOLDER,
        path=input_data_uri
    ),
    pipeline_config_type="ecomm",
    pipeline_incremental=False
)

# Set metadata
pipeline_job.display_name = "PA Pipeline - Import Fix"
pipeline_job.experiment_name = "personal_agendas_experiment"

# Submit
try:
    print("\nSubmitting pipeline...")
    submitted_job = ml_client.jobs.create_or_update(pipeline_job)
    
    print(f"\n✅ SUCCESS!")
    print(f"Job Name: {submitted_job.name}")
    print(f"Monitor at: {submitted_job.studio_url}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTrying simpler test job...")
    
    # Try a minimal test
    try:
        test_job = command(
            code=str(project_root),
            command="ls -la && echo 'Test successful'",
            environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
            compute="cpu-cluster"
        )
        test_result = ml_client.jobs.create_or_update(test_job)
        print(f"Test job submitted: {test_result.studio_url}")
    except Exception as e2:
        print(f"Test job also failed: {e2}")