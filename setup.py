#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []

test_requirements = []

setup(
    author="Zhuoran Qiao",
    author_email="zqiao@caltech.edu",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="Score-based Dynamic-backbone Protein-ligand Structure Prediction",
    entry_points={
        "console_scripts": [
            "neuralplexer-train=neuralplexer.cli:main",
            "neuralplexer-inference=neuralplexer.inference:main",
        ],
    },
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    package_data={
        "neuralplexer": [
            "data/*.csv",
            "data/*.pdb", 
            "data/*.json",
            "data/*.txt",
            "data/chemical/*.pdb",
            "data/chemical/*.json",
            "data/chemical/*.txt",
        ],
    },
    keywords="neuralplexer",
    name="neuralplexer",
    packages=find_packages(include=["neuralplexer", "neuralplexer.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/zrqiao/GPLC",
    version="0.1.0",
    zip_safe=False,
)
