import os
from setuptools import setup, find_packages

long_description = ""
if os.path.exists("README.md"):
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="microvectordb",
    version="0.7.0",
    description="A high-performance, low-memory, persistent vector database for Python, RAG, and AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ahmet GÜN",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "microvectordb-server=microvectordb.api.server:main",
        ],
    },
    extras_require={
        "hnsw": ["hnswlib>=0.7.0"],
        "fast": ["orjson>=3.9.0"],
        "server": ["fastapi>=0.110.0", "uvicorn>=0.27.0", "pydantic>=2.0.0"],
        "all": ["hnswlib>=0.7.0", "orjson>=3.9.0", "fastapi>=0.110.0", "uvicorn>=0.27.0", "pydantic>=2.0.0"],
        "bench": ["psutil>=5.9.0", "faiss-cpu>=1.7.4"],
        "dev": ["pytest>=7.0.0", "psutil>=5.9.0", "httpx>=0.27.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
