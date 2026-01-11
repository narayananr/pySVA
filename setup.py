from setuptools import setup, find_packages

setup(
    name="pysva",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
    ],
    author="Narayanan Raghupathy",
    description="Surrogate Variable Analysis in Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
