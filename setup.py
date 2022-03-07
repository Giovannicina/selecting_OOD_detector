import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='selecting_ood_detector',
    version='0.0.0',
    author='Pacmed BV',
    description='Out-of-distribution detection for Tabular Data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Giovannicina/selecting_OOD_detector',
    license='MIT',
    packages=['toolbox'],
    install_requires=['requests'],
)