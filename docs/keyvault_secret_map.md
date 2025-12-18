# Key Vault → Environment Variable Map

This document captures every secret the Personal Agendas pipeline expects to read from Azure Key Vault and how each value is consumed. Use it to verify that the vault contains all required entries before running the local or Azure ML pipelines.

## Automatically Pulled by `KeyVaultManager`
The helper in [PA/utils/keyvault_utils.py](PA/utils/keyvault_utils.py) fetches the following secret names (case-insensitive) and writes them into `PA/keys/.env` as uppercase environment variables.

| Key Vault secret name | Env var written | Consumers | Purpose |
| --- | --- | --- | --- |
| `openai-api-key` | `OPENAI_API_KEY` | Registration/session processors, embeddings, recommendations (LangChain/OpenAI clients) | LLM key |
| `azure-api-key` | `AZURE_API_KEY` | Same as above when using Azure OpenAI | Azure OpenAI key |
| `azure-endpoint` | `AZURE_ENDPOINT` | Same | Azure OpenAI endpoint URL |
| `azure-deployment` | `AZURE_DEPLOYMENT` | Same | Azure OpenAI deployment name |
| `azure-api-version` | `AZURE_API_VERSION` | Same | API version used by Azure OpenAI clients |
| `neo4j-uri` or `NEO4J-URI` | `NEO4J_URI` (plus env-specific copy) | All Neo4j processors, embeddings, recommendations | Graph endpoint |
| `neo4j-username` or `NEO4J-USERNAME` | `NEO4J_USERNAME` | Same | Graph username |
| `neo4j-password` or `NEO4J-PASSWORD` | `NEO4J_PASSWORD` (plus env-specific copy) | Same | Graph password |
| `databricks-token` | `DATABRICKS_TOKEN` | MLflow integration when pointing to Databricks | PAT |
| `databricks-hosts` | `DATABRICKS_HOSTS` (map to `DATABRICKS_HOST` if needed) | Same | Workspace URL(s) |
| `mlflow-tracking-uri` | `MLFLOW_TRACKING_URI` | Pipelines + notebooks for experiment logging | Tracking endpoint |
| `mlflow-registry-uri` | `MLFLOW_REGISTRY_URI` | Same | Registry endpoint |
| `mlflow-experiment-id` | `MLFLOW_EXPERIMENT_ID` | Same | Default experiment path |

> **Note:** `apply_neo4j_credentials()` (see [azureml_pipeline/neo4j_env_utils.py](azureml_pipeline/neo4j_env_utils.py)) also persists the Neo4j secrets into `.env` with environment-specific suffixes (`NEO4J_URI_PROD`, etc.) whenever it refreshes credentials.

## Telemetry / Application Insights
`configure_app_insights()` looks for the following environment variables but they are **not** fetched from Key Vault yet. Add them to the vault if you want telemetry to hydrate automatically:

| Recommended KV secret name | Env var(s) to populate | Consumers |
| --- | --- | --- |
| `applicationinsights-connection-string` | `APPLICATIONINSIGHTS_CONNECTION_STRING` (aliases: `AZURE_APPINSIGHTS_CONNECTION_STRING`, `APPINSIGHTS_CONNECTION_STRING`) | Local pipeline entrypoint, Azure ML steps 1–4, Streamlit frontend |

If you add this secret, extend `KeyVaultManager.get_all_secrets()` to include the new name so it’s written into `.env` alongside the others.

## Additional Environment Metadata
These values are currently sourced from the developer/notebook `.env` file or pipeline parameters. Store them in Key Vault only if you want centralized management:

| Env var | Usage | Suggested Key Vault secret name |
| --- | --- | --- |
| `AZURE_CLIENT_ID` | Service principal client ID for Key Vault / Azure ML auth | `azure-client-id` |
| `AZURE_CLIENT_SECRET` | Service principal secret | `azure-client-secret` |
| `AZURE_TENANT_ID` | Tenant ID used by the SPN | `azure-tenant-id` |
| `SUBSCRIPTION_ID` | Azure subscription targeting each notebook/pipeline run | `azure-subscription-id` |
| `RESOURCE_GROUP` | Resource group containing the Azure ML workspace | `azure-resource-group` |
| `AZUREML_WORKSPACE_NAME` | Azure ML workspace name | `azureml-workspace-name` |
| `NEO4J_URI_DEV`, `NEO4J_URI_TEST`, `NEO4J_URI_PROD` | Environment-specific Neo4j Aura URIs used by notebooks and fallbacks | `neo4j-uri-dev`, `neo4j-uri-test`, `neo4j-uri-prod` |
| `NEO4J_PASSWORD_DEV`, `NEO4J_PASSWORD_TEST`, `NEO4J_PASSWORD_PROD` | Matching passwords for each Neo4j environment | `neo4j-password-dev`, `neo4j-password-test`, `neo4j-password-prod` |
| `AURA_INSTANCEID`, `AURA_INSTANCENAME` (optional) | Aura metadata used by tooling scripts | `aura-instanceid`, `aura-instancename` |
| Any other service-specific keys (e.g., `OCP_APIM_SUBSCRIPTION_KEY`) | Depends on integration | follow kebab-case, e.g., `ocp-apim-subscription-key` |

Keeping this table up to date ensures both the local scripts and every Azure ML pipeline step load identical secrets, so behavior stays consistent across environments.
