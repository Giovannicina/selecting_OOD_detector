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
    version="0.1.3",

    # package_dir={'': 'selecting_OOD_detector'},
    py_modules=['selecting_OOD_detector'],
    packages=['selecting_OOD_detector'],
    author='Pacmed BV',
    url="https://github.com/Giovannicina/selecting_OOD_detector",

    python_requires=">=3.6",
)
