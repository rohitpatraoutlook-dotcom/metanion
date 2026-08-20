from setuptools import setup, find_packages

setup(
    name="metanion",
    version="4.0.0",
    author="Metanion Community",
    description="Zero-Weight Symbolic Tensor Engine with Knowledge Base",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rohitpatraoutlook-dotcom/metanion",
    packages=find_packages(),
    include_package_data=True,
    package_data={'metanion': ['knowledge_base/*.pkl.gz']},
    install_requires=["numpy>=1.19.0"],
    python_requires=">=3.8",
)
