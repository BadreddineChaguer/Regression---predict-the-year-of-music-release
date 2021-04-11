"""
@author: CHAGUER Badreddine
"""


import setuptools


setuptools.setup(
    name = "Prediction_years_of_songs",
    version="0.0.1",
    author="CHAGUER badreddine",
    author_email="badreddine.chaguer@gmail.com",
    description="A small example package",
    long_description=open('README.rst').read(),
    url="https://github.com/BadreddineDS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python",
        "Natural Language :: French",
        "Programming Language :: Python :: 3.7.6",
    ],
    python_requires='>=3.6',
)