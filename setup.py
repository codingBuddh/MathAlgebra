from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mathalgebra",
    version="0.1.0",
    author="Aman",
    author_email="soniaman809@gmail.com",
    description="A linear algebra library for mathematical operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/codingBuddh/MathAlgebra/",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
) 