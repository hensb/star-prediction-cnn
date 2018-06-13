from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['cloudpickle', 'dask[complete]', 'gcsfs', 'keras', 'numpy', 'pandas', 'toolz', 'google-cloud-storage']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)