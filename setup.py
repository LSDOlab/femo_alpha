import os
import codecs
from setuptools import setup, find_packages

base_dir = os.path.abspath(os.path.dirname(__file__))
def read(fname):
    return codecs.open(os.path.join(base_dir, fname), encoding="utf-8").read()

setup(
    name='femo_alpha',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/LSDOlab/femo_alpha',
    license='GNU LGPLv3',
    author='Ru Xiang',
    author_email='rxiang@ucsd.edu',
    description="Finite Element for Multidisciplinary Optimization",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
)
