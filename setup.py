#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

from setuptools import setup

requirements = [
    'tensorflow>=1.10.0',
    'numpy>=1.10.0',
    'pytest>=3.5.0',
]

setup(
    name='tensorcheck',
    version="0.0.1",
    author="Erik Bodin, Andrew Lawrence",
    author_email="mail@erikbodin.com",
    description="Tensor validation using TensorFlow",
    license="MIT",
    keywords="machine-learning gaussian-processes kernels tensorflow",
    url="https://github.com/bodin-e/tensorcheck",
    packages=["tensorcheck"],
    install_requires=requirements,
    test_suite='tests',
)