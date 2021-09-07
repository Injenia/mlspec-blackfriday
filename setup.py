from setuptools import find_packages
from setuptools import setup

setup(
    name='mlspec-blackfriday',
    version='0.1',
    url='https://bitbucket.org/injenia/mlspec-blackfriday/src/main/',
    license='',
    author='Injenia',
    install_requires=[
        'tensorflow==2.4.0',
        'tensorflow-recommenders==0.4.0',
        'scann==1.2.1',
    ],
    packages=find_packages(),
    author_email='',
    description=''
)
