# The find_packages methodconducts a static analysis of libraries included in this package. 
# It forces install if required libraries are not already present.

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ML_prod_utils',
    version='0.1.0',
    description='Template API interface to deploy machine learning code',
    long_description=readme,
    author='Gurnimrat Sidhu',
    author_email='gurnimratsidhu@gmail.com',
    url='https://github.com/gksighu/literate-octo-pancake',
    license=license,
    # The below line can be edited to directly list the required libraries, e.g. packages = scikit-learn, pandas, numpy
    packages=find_packages(exclude=('tests'))
)
