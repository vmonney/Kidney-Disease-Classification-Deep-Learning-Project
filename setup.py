"""Setup script for the Kidney-Disease-Classification-Deep-Learning-Project package.

This script uses setuptools to define the package
metadata and dependencies, and to install the package.

The package metadata includes the package name, version, author,
author email, description, long description, and project URLs.

The package dependencies are defined in the requirements.txt file.

To install the package, run `python setup.py install` in the terminal.

"""

from pathlib import Path

import setuptools

with Path("README.md").open(encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Kidney-Disease-Classification-Deep-Learning-Project"
AUTHOR_USER_NAME = "vmonney"
SRC_REPO = "cnn_classifier"
AUTHOR_EMAIL = "monney.valentin@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
