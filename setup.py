from setuptools import setup

__version__ = '1.0.2'

setup(
    name = 'ds_tools',
    version = __version__,
    description = 'A python library for data science tasks.',
    url = 'git+http://github.com/AdmiralWen/ds_tools.git',
    author = 'Brandon Wen',
    license = 'MIT',
    packages = ['ds_tools'],
    zip_safe = False
)