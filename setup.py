import os
from setuptools import setup, find_packages

long_description = ""
if os.path.exists("README.md"):
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="microvectordb",
    version="0.2.0",
    description="A lightweight, fast, and persistent vector database with ChromaDB/FAISS-like API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Huneyn Kaya",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
