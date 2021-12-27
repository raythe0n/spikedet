from setuptools import find_packages, setup

setup(
    name='spikedet',
    packages=find_packages(),
    version='0.0.1',
    description='Spike detector ML',
    author='Andrey',
    author_email="aaaaaaaaa@gmail.com",
    license='Commercial',
    install_requires=[
        'numpy',
        'torch'
    ],
)
