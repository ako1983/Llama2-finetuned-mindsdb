from setuptools import find_packages, setup

setup(
    name="my_training_package",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers==4.37.0",
        "torch==2.1.0",
        "peft",
        "datasets"
    ],
)
