# mrp7pred

A machine learning pipeline to predict putative MRP7 ligands

## Overview
--------
![](./mrp7_small.gif)

_A dancing MRP7 protein (Homology model)._

MRP7, also named ATP-binding cassette transporter C10 (ABCC10), was first
discovered in 2003 and has been proved to mediate multidrug resistance in
cancer cells.

## Install Java and JDK for cdk fingerprints and descriptors
--------
- [Install Java](https://www.java.com/en/download/help/mac_install.html)

- [Install JDK](https://docs.oracle.com/en/java/javase/15/install/installation-jdk-macos.html#GUID-2FE451B0-9572-4E38-A1A5-568B77B146DE)


## Installation
--------
```bash
git clone https://github.com/jingquan-wang/mrp7pred.git
cd mrp7pred
conda env create -f environment.yml
conda activate mrp7pred
pip install -e .
pip install -r requirements.txt
```

## Demo
--------
Check `demo.ipynb`
