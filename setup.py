from setuptools import setup, find_packages

setup(
    name="mrp7pred",
    version="1.0",
    description="A machine learning pipeline to predict putative MRP7 ligands",
    author="Jing-Quan Wang",
    author_email="jq.wang1214@gmail.com",
    license="GPL",
    packages=["mrp7pred"],
    package_dir={'pychem':'src/mrp7pred'}    
)