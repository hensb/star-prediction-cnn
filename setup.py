from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['cloudpickle==0.5.3', 'dask[complete]==0.18.1', 'gcsfs==0.1.0', 'keras==2.2.0', 'numpy==1.14.4',
                     'pandas==0.23.1',
                     'requests==2.18.4', 'toolz==0.9.0',
                     'urllib3', 'google-cloud-storage==1.3.2']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
