"""
Azure Key Vault utilities for Personal Agendas pipeline.
"""
import os
import logging
from typing import Dict, Optional
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import ResourceNotFoundError

logger = logging.getLogger(__name__)


class KeyVaultManager:
    """Manage secrets from Azure Key Vault."""
    
    def __init__(self, keyvault_name: Optional[str] = None):
        """
        Initialize Key Vault manager.
        
        Args:
            keyvault_name: Name of the Key Vault (or from environment)
        """
        self.keyvault_name = keyvault_name or os.environ.get("KEYVAULT_NAME", "pa-keyvault-dev-01")
        self.keyvault_url = f"https://{self.keyvault_name}.vault.azure.net"
        self.client = None
        self._secrets_cache = {}
        
    def _get_client(self) -> SecretClient:
        """Get or create Key Vault client."""
        if not self.client:
            try:
                # Try managed identity first (for Azure ML compute)
                credential = ManagedIdentityCredential()
                self.client = SecretClient(self.keyvault_url, credential)
                logger.info("Using Managed Identity for Key Vault access")
            except Exception:
                # Fallback to default credential (for local development)
                credential = DefaultAzureCredential()
                self.client = SecretClient(self.keyvault_url, credential)
                logger.info("Using Default Credential for Key Vault access")
        return self.client
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """
        Get a secret from Key Vault.
        
        Args:
            secret_name: Name of the secret
            
        Returns:
            Secret value or None if not found
        """
        # Check cache first
        if secret_name in self._secrets_cache:
            return self._secrets_cache[secret_name]
        
        try:
            client = self._get_client()
            secret = client.get_secret(secret_name)
            self._secrets_cache[secret_name] = secret.value
            return secret.value
        except ResourceNotFoundError:
            logger.warning(f"Secret '{secret_name}' not found in Key Vault")
            return None
        except Exception as e:
            logger.error(f"Error retrieving secret '{secret_name}': {e}")
            return None
    
    def get_all_secrets(self) -> Dict[str, str]:
        """
        Get all PA-related secrets from Key Vault.
        
        Returns:
            Dictionary of secret names and values
        """
        secret_names = [
            "openai-api-key",
            "azure-api-key", 
            "azure-endpoint",
            "azure-deployment",
            "azure-api-version",
            "neo4j-uri",
            "neo4j-username",
            "neo4j-password",
            "databricks-token",
            "databricks-hosts",
            "mlflow-tracking-uri",
            "mlflow-registry-uri",
            "mlflow-experiment-id"
        ]
        
        secrets = {}
        for name in secret_names:
            value = self.get_secret(name)
            if value:
                # Convert to environment variable format
                env_name = name.upper().replace("-", "_")
                secrets[env_name] = value
        
        return secrets
    
    def create_env_file(self, output_path: str = "PA/keys/.env") -> bool:
        """
        Create a .env file from Key Vault secrets.
        
        Args:
            output_path: Path to save the .env file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            secrets = self.get_all_secrets()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write to .env file
            with open(output_path, 'w') as f:
                f.write("# Secrets loaded from Azure Key Vault\n")
                f.write(f"# Key Vault: {self.keyvault_name}\n\n")
                
                for key, value in secrets.items():
                    f.write(f"{key}={value}\n")
            
            logger.info(f"Created .env file at {output_path} with {len(secrets)} secrets")
            return True
            
        except Exception as e:
            logger.error(f"Error creating .env file: {e}")
            return False


def load_secrets_from_keyvault(keyvault_name: Optional[str] = None) -> Dict[str, str]:
    """
    Convenience function to load secrets from Key Vault.
    
    Args:
        keyvault_name: Optional Key Vault name
        
    Returns:
        Dictionary of secrets
    """
    manager = KeyVaultManager(keyvault_name)
    return manager.get_all_secrets()


def ensure_env_file(keyvault_name: Optional[str] = None, env_path: str = "PA/keys/.env") -> bool:
    """
    Ensure .env file exists by creating it from Key Vault if needed.
    
    Args:
        keyvault_name: Optional Key Vault name
        env_path: Path to .env file
        
    Returns:
        True if .env file exists or was created successfully
    """
    if os.path.exists(env_path):
        logger.info(f".env file already exists at {env_path}")
        return True
    
    logger.info(f".env file not found, creating from Key Vault...")
    manager = KeyVaultManager(keyvault_name)
    return manager.create_env_file(env_path)