from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='swiss_army_tensorboard',
    version='0.0.1',
    url='gaborvecsei.com',
    author='Gabor Vecsei',
    install_requires=requirements,
    packages=find_packages(),
)
