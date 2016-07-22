from setuptools import setup, find_packages

setup(
    name='pymot',
    version='0.0.0',
    install_requires=['numpy'],
    packages=find_packages(),
    test_requires=['nose']
)
