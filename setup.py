from setuptools import setup

setup(
    name = 'ds_tools',
    version = open("ds_tools/_version.py").readlines()[-1].split()[-1].strip("\"'"),
    description = 'A python library for data science tasks.',
    url = 'git+http://github.com/AdmiralWen/ds_tools.git',
    author = 'Brandon Wen',
    license = 'MIT',
    packages = ['ds_tools'],
    zip_safe = False
)