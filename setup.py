from setuptools import setup, find_packages


setup(
    name="mrp7pred",
    version="1.0",
    description="A machine learning pipeline to predict putative MRP7 ligands",
    author="Jing-Quan Wang",
    author_email="jq.wang1214@gmail.com",
    license="GPL",
    packages=find_packages(),
    # package_dir={'pychem':'mrp7pred'}
    install_requires=[
        "scikit-learn",
        "seaborn",
        "tqdm",
        "xgboost",
        "pandas",
        "numpy",
        "PyFingerprint @ git+https://github.com/hcji/PyFingerprint@master",
        "jpype1",
    ],
)