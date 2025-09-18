#!/usr/bin/env python3
"""
Setup script for MeerTOD: Time-Ordered Data Simulator for Radio Astronomy

This package provides comprehensive tools for simulating time-ordered data (TOD) 
from radio astronomy observations, specifically designed for MeerKAT telescope 
but adaptable to other radio telescopes.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements from requirements.txt
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="meerTOD",
    version="1.0.0",
    author="Zheng Zhang",
    author_email="zheng.zhang@manchester.ac.uk",
    description="Time-Ordered Data Simulator for MeerKLASS",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/zzhang0123/meerTOD",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.800',
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
            'jupyter>=1.0',
            'matplotlib>=3.3.0',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
            'jupyter>=1.0',
            'matplotlib>=3.3.0',
        ],
        'notebooks': [
            'jupyter>=1.0',
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
        ]
    },
    include_package_data=True,
    package_data={
        'meerTOD': ['data/*', 'examples/*'],
    },
    keywords=[
        'time-ordered data', 'simulation', 'MeerKAT',
        'HEALPix', 'beam patterns', 'sky models', 'noise modeling'
    ],
    project_urls={
        'Bug Reports': 'https://github.com/zzhang0123/meerTOD/issues',
        'Source': 'https://github.com/zzhang0123/meerTOD',
        'Documentation': 'https://meertod.readthedocs.io/',
    },
)
