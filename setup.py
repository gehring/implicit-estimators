
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
    name='imprl',
    version='0.0.1',
    python_requires='==3.*,>=3.8.0',
    packages=find_packages(
        exclude=['additional'],
    ),
)
