"""
Setup script for Metanion - A Zero-Weight Symbolic Tensor Engine
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
with open("metanion/__init__.py", "r") as f:
    content = f.read()
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    version = version_match.group(1) if version_match else "0.1.0"

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="metanion",
    version=version,
    author="Rohit Patra",
    author_email="rohitpatra@outlook.com",
    description="A zero-weight symbolic tensor engine that learns mathematical expressions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rohitpatraoutlook-dotcom/metanion",
    project_urls={
        "Bug Tracker": "https://github.com/rohitpatraoutlook-dotcom/metanion/issues",
        "Documentation": "https://github.com/rohitpatraoutlook-dotcom/metanion#readme",
        "Source Code": "https://github.com/rohitpatraoutlook-dotcom/metanion",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "cupy>=12.0.0",
            "numba>=0.57.0",
        ],
        "visualization": [
            "graphviz>=0.20.0",
            "matplotlib>=3.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "metanion=metanion.metanion_engine:main",
        ],
    },
    keywords=[
        "symbolic-regression",
        "genetic-programming",
        "tensor-engine",
        "machine-learning",
        "symbolic-ai",
        "differentiable-programming",
    ],
)