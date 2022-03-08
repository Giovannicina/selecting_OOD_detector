# import setuptools
# from setuptools import setup, find_packages
#
# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()
#
# setuptools.setup(
#     name='selecting_ood_detector',
#     version='0.1',
#     author='Pacmed BV',
#     description='Out-of-distribution detection for Tabular Data',
#     long_description=long_description,
#
#     long_description_content_type="text/markdown",
#     url='https://github.com/Giovannicina/selecting_OOD_detector',
#     license='MIT',
#     # packages=['selecting_ood_detector'],
#     packages=find_packages(),
#     install_requires=['requests'],
# )

import codecs
from setuptools import setup, find_packages

with codecs.open("README.md", encoding="utf-8") as f:
    README = f.read()


setup(
    name="selecting_ood_detector",
    description='Out-of-distribution detection for Tabular Data',
    long_description=README,
    long_description_content_type="text/markdown",
    version="0.1.2",
    packages=find_packages(),
    author='Pacmed BV',
    url="https://github.com/y0ast/DUE",
    # author_email="joost.van.amersfoort@cs.ox.ac.uk",
    # install_requires=["gpytorch>=1.2.1", "torch", "scikit-learn"],
    python_requires=">=3.6",
)