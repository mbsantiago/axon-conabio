# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

version = {}
with open("./axon_conabio/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='axon-conabio',
    version=version['__version__'],
    description='Conjunto de herramientas para crear, entrenar y evaluar modelos.',
    author='Santiago Martínez, Everardo Robredo',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email='santiago.mbal@gmail.com',
    url='https://github.com/mbsantiago/axon-conabio',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'tqdm',
        'six',
        'click',
        'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'axon=axon_conabio.management.commands:main',
        ]
    }
)
