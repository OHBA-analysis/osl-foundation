from setuptools import setup, find_packages


setup(
    name='osl_foundation',
    version='0.0.1',
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_dir={'osl_foundation': 'osl_foundation'},
)

