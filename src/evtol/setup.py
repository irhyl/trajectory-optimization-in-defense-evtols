#!/usr/bin/env python3
"""
Setup script for eVTOL Trajectory Optimization System.

This setup file defines project metadata, dependencies, and installation
configuration for academic distribution and reproducibility.

Usage:
    pip install -e .  # Editable installation
    pip install -e ".[dev]"  # With development dependencies
    pip install -e ".[docs]"  # With documentation dependencies
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
README_FILE = Path(__file__).parent / "README.md"
long_description = README_FILE.read_text(encoding="utf-8") if README_FILE.exists() else ""

setup(
    # Package metadata
    name="evtol-trajectory-optimization",
    version="0.1.0",
    description="Multi-objective trajectory optimization system for eVTOL aircraft in defense applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="IISc Research Team",
    author_email="research@iisc.ac.in",
    organization="Indian Institute of Science",
    url="https://github.com/irhyl/trajectory-optimization-in-defense-evtols",
    
    # Project classification
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Aerospace",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for discovery
    keywords=[
        "trajectory optimization",
        "eVTOL",
        "electric aircraft",
        "multi-objective optimization",
        "path planning",
        "defense applications",
        "research platform",
    ],
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    include_package_data=True,
    
    # Core dependencies (must match requirements.txt)
    install_requires=[
        "numpy==1.24.3",
        "scipy==1.10.1",
        "pandas==2.0.3",
        "networkx==3.1",
        "shapely==2.0.1",
        "pyproj==3.5.0",
        "pydantic==2.5.0",
        "pyyaml==6.0.1",
        "loguru==0.7.2",
        "deap==1.4.1",
        "casadi==3.5.5",
        "pyomo==6.4.4",
        "scikit-learn==1.3.2",
        "plotly==5.17.0",
        "streamlit==1.28.1",
        "pandas==2.0.3",
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
    ],
    
    # Optional dependency groups
    extras_require={
        "dev": [
            "pytest==7.4.3",
            "pytest-cov==4.1.0",
            "pytest-benchmark==4.0.0",
            "black==24.3.0",
            "ruff==0.6.2",
            "mypy==1.10.0",
            "pylint==3.0.3",
            "pre-commit==3.7.1",
        ],
        "docs": [
            "sphinx==7.0.1",
            "sphinx-rtd-theme==1.3.0",
            "sphinx-autodoc-typehints==1.24.0",
            "myst-parser==1.0.0",
        ],
        "ml": [
            "tensorflow==2.14.0",
            "torch==2.1.0",
        ],
        "full": [
            # Include everything
            "pytest==7.4.3",
            "pytest-cov==4.1.0",
            "sphinx==7.0.1",
            "sphinx-rtd-theme==1.3.0",
            "tensorflow==2.14.0",
            "torch==2.1.0",
        ],
    },
    
    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "evtol-planner=evtol.cli.main:planner",
            "evtol-simulator=evtol.cli.main:simulator",
            "evtol-analyze=evtol.cli.main:analyzer",
        ],
    },
    
    # Project URLs
    project_urls={
        "Bug Tracker": "https://github.com/irhyl/trajectory-optimization-in-defense-evtols/issues",
        "Documentation": "https://trajectory-optimization-in-defense-evtols.readthedocs.io",
        "Source Code": "https://github.com/irhyl/trajectory-optimization-in-defense-evtols",
        "Papers": "https://github.com/irhyl/trajectory-optimization-in-defense-evtols/tree/main/docs",
    },
    
    # Additional metadata
    zip_safe=False,
)
