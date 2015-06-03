#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Andrew J. Lambert, andy@andyroid.co.uk'


from setuptools import setup, find_packages


setup(
    name="PyGFNN",
    version="0.0.1",
    description="PyGFNN is a GFNN extension for PyBrain.",
    license="BSD",
    keywords="Neural Networks Machine Learning",
    url="http://pybrain.org",
    packages=find_packages(exclude=['examples', 'docs']),
    include_package_data=True,
    test_suite='pygfnn.tests.runtests.make_test_suite',
    package_data={'pygfnn': ['rl/environments/ode/models/*.xode']},
    install_requires = ["pybrain"],
)