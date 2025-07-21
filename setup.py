#!/usr/bin/env python3
"""
Setup script for IntroStyle feature extraction.
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="introstyle",
    version="1.0.0",
    author="IntroStyle Team",
    author_email="contact@introstyle.ai",
    description="Introspective Style Attribution using Diffusion Features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cuixing100876/InstaStyle",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipywidgets>=7.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "introstyle-extract=extract_features:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
