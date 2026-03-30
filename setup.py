#!/usr/bin/env python
"""
AQDnet setup script
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='aqdnet',
    version='0.1.0',
    author='AQDnet Authors',
    description='Deep Neural Network for Protein-Ligand Docking and Scoring',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/koji11235/AQDnet',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    # Note: Dependencies are already provided in environment.yml and Docker image
    install_requires=[],
    entry_points={
        'console_scripts': [
            'aqdnet=aqdnet_cli:main',
        ],
    },
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
