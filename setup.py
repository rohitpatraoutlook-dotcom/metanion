from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
with open("metanion/__init__.py", "r") as f:
    content = f.read()
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    version = version_match.group(1) if version_match else "3.0.0"

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="metanion",
    version=version,
    author="Rohit Patra",
    author_email="rohitpatra@outlook.com",
    description="A zero-weight symbolic tensor engine for interpretable machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rohitpatraoutlook-dotcom/metanion",
    packages=find_packages(include=['metanion', 'metanion.*', 'research']),
    include_package_data=True,
    package_data={
        'research': ['*.py'],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
    ],
)
