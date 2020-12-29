from openeye import oechem

mol = oechem.OEGraphMol()

if oechem.OESmilesToMol(mol, "c1ccccc1"):
    smi = oechem.OECreateCanSmiString(mol)
    print("Canonical SMILES is %s\n" % smi)

else:
    print("SMILES string was invalid!\n")
