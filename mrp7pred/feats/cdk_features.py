from mrp7pred.cinfony_py3 import cdk

print(cdk.aromaticity)
mol = cdk.readstring("smi", "CNC")
print (mol.molwt)

