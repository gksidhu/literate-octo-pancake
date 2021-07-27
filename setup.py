from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ML_prod_utils',
    version='0.1.0',
    description='Template to standardize production ML references',
    long_description=readme,
    author='Gurnimrat Sidhu',
    author_email='gurnimratsidhu@gmail.com',
    url='https://github.com/gksighu/literate-octo-pancake',
    license=license,
    packages=find_packages(exclude=('tests'))
)
