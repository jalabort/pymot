from setuptools import setup, find_packages


setup(
    name='pymot',
    version='0.0.0',
    install_requires=['numpy', 'cython', 'tqdm'],
    dependency_links=[
        'http://github.com/jfrelinger/cython-munkres-wrapper/master#egg=munkres'],
    packages=find_packages(),
    tests_require=['nose']
)
