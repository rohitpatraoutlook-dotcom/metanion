from setuptools import setup, find_packages

setup(
    name="metanion",
    version="3.0.0",
    author="Rohit Patra",
    author_email="rohitpatra@outlook.com",
    description="A zero-weight symbolic tensor engine",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rohitpatraoutlook-dotcom/metanion",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    install_requires=["numpy>=1.19.0"],
)
