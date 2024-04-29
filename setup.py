
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ssmjax',
    version='0.1',
    description='',
    author='AK',
    author_email='',
    url='',
    packages=find_packages(),
    install_requires=requirements,
)
