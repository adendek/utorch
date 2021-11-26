from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='utorch',
   version='0.1',
   description='Understandable implementation of PyTorch',
   license="MIT",
   long_description=long_description,
   long_description_content_type='text/markdown',
   author='Adam Dendek',
   author_email='arath',
   packages=find_packages(), 
   install_requires=['wheel', 'numpy', "torch"],
   url="https://github.com/adendek/utorch",
   python_requires='>=3.6',
)