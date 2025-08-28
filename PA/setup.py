from setuptools import setup, find_packages

setup(
    name="vet_conference_pipeline",
    version="1.0.0",
    description="Data processing pipeline Personal Agendas",
    author="Conference Data Team",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "neo4j>=4.4.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.19.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
)
