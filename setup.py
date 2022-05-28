#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ConfSolv",
    version="0.0.1",
    author="Lagnajit Pattanaik",
    author_email="lagnajit@mit.com",
    description="Predicting solution free energies for 3D conformers with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PattanaikL/conf_solv",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry"
    ],
    license="MIT License",
    python_requires='>=3.7',
)
