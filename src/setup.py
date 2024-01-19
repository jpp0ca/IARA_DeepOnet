from setuptools import find_packages, setup

setup(
    name='iara',
    packages=find_packages(),
    version='0.1',
    description='package to access and train against the iara dataset',
    author='Fábio Oliveira, Natanael Junior',
    license='Apache-2.0 License',
    install_requires=[],
    package_data={'iara': ['dataset_info/*.csv']},
)