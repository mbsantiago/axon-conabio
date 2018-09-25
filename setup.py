#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='axon-conabio',
    version='0.0.1',
    description='Conjunto de herramientas para crear, entrenar y evaluar modelos.', 
    author='Santiago Martinez, Everardo Robredo',
    long_description=long_description,
    long_description_content_type="text/markdown", 
    author_email='santiago.mbal@gmail.com',
    url='https://github.com/mbsantiago/axon-conabio',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
