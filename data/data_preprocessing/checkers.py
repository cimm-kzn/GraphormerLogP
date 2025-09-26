from rdkit import Chem
from chython.utils.rdkit import to_rdkit_molecule
from chython import smiles
import numpy as np


class Molecule:
    def __init__(self, smi):
        self.smi = smi
        self.molecule = None
        self.smi_canon = None
        self.smi_non_stereo = None
        self.smi_non_radical = None
        self.allowed_atoms = ['C', 'Na', 'K', 'S', 'N', 'Cl', 'Br', 'I']
        self.valid = False
        self.has_radicals = False
        self.has_stereo = False
        self.invalid_reason = None

    def check_metals(self):
        brutto = set([i.GetSymbol() for i in self.molecule.GetAtoms()])
        for _atom in brutto:
            if _atom in self.allowed_atoms:
                return True
        return False

    def simple_checker(self):
        if len(self.smi.split('.')) != 1:
            self.invalid_reason = "Multiple fragments (contains '.')"
            return False
        self.molecule = Chem.MolFromSmiles(self.smi)
        if self.molecule:
            if self.check_metals():
                return True
            else:
                self.invalid_reason = "No allowed atoms"
                return False
        self.invalid_reason = "RDKit parsing failed"
        return False

    def _remove_stereo(self):
        if Chem.FindMolChiralCenters(self.molecule, includeUnassigned=True):
            self.has_stereo = True
        Chem.RemoveStereochemistry(self.molecule)
        [a.SetAtomMapNum(0) for a in self.molecule.GetAtoms()]
        return Chem.MolToSmiles(self.molecule, canonical=True)

    def _remove_radicals(self):
        for a in self.molecule.GetAtoms():
            if a.GetNumRadicalElectrons() == 1 and a.GetFormalCharge() == 1:
                self.has_radicals = True
                a.SetNumRadicalElectrons(0)
                a.SetFormalCharge(0)
    
    def prepare(self):
        if self.simple_checker():
            self._remove_radicals()
            canon_smi = self._remove_stereo()
            return canon_smi
