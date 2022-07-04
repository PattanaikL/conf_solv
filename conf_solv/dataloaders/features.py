from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem import AllChem, Crippen

from argparse import Namespace
from typing import List, Union

SOLVENTS = {'scf': 'scf',
            'Acetat': 'CC(=O)[O-]',
            'Acetic-acid': 'CC(=O)O',
            'Aceton': 'CC(=O)C',
            'Acetonitrile': 'CC#N',
            'Ammonia': 'N',
            'Ammonium': '[NH4+]',
            'Benzene': 'c1ccccc1',
            'Benzoat': '[O-]C(=O)c1ccccc1',
            'Benzylacetat': 'CC(=O)OCc1ccccc1',
            'Butanon': 'CCC(=O)C',
            'Chloride': '[Cl-]',
            'Chloroform': 'C(Cl)(Cl)Cl',
            'Cyclohexan': 'C1CCCCC1',
            'Di-2-butylamin': 'CC[C@H](C)N[C@H](C)CC',
            'Dichlormethan': 'C(Cl)Cl',
            'Diethanolamin': 'OCCNCCO',
            'Diethanolammonium': 'OCC[NH2+]CCO',
            'Diethylenamin': 'CCNCC',
            'Diethylenammonium': 'CC[NH2+]CC',
            'Diethylether': 'CCOCC',
            'Dioctylether': 'CCCCCCCCOCCCCCCC',
            'DMA': 'CC(=O)N(C)C',
            'DMF': 'CN(C)C=O',
            'DMSO': 'CS(=O)C',
            'EC': 'C1COC(=O)O1',
            'EMC': 'CCOC(=O)OC',
            'Ethanol': 'CCO',
            'Ethylacetat': 'CCOC(=O)C',
            'Ethylenamin': 'CCN',
            'Ethylenammonium': 'CC[NH3+]',
            'Ethylenglykol': 'OCCO',
            'Formiat': 'C(=O)[O-]',
            'Formic-acid': 'C(=O)O',
            'g-Butyrolacton': 'O=C1CCCO1',
            'Glycerin': 'OCC(O)CO',
            'H2O': 'O',
            'H2SO4': 'O=S(=O)(O)O',
            'Hexafluorbenzol': 'Fc1c(F)c(F)c(F)c(F)c1F',
            'Isooctane': 'CC(C)CC(C)(C)C',
            'Isopropanol': 'CC(O)C',
            'Methanolat': 'C[O-]',
            'n-Hexan': 'CCCCCC',
            'Nonandecanol': 'CCCCCCCCCCCCCCCCCCCO',
            'Octanol': 'OCCCCCCCC',
            'o-Dichlorbenzol': 'Clc1ccccc1Cl',
            'Perfluoro-hexan': 'C(C(C(C(F)(F)F)(F)F)(F)F)(C(C(F)(F)F)(F)F)(F)F',
            'Propylenglykol': 'C[C@@H](O)CO',
            'Tetraethylenammonium': 'CC[N+](CC)(CC)CC',
            'THF': 'O1CCCC1',
            'Toluol': 'Cc1ccccc1',
            'Tributylphosphat': 'O=P(OCCCC)(OCCCC)OCCCC',
            'Triethanolamin': 'OCCN(CCO)CCO',
            'Triethanolammonium': 'OCC[NH+](CCO)CCO',
            'Triethylenamin': 'CCN(CC)CC',
            'Triethylenammonium': 'CC[NH+](CC)CC',
            'Triglyme': 'COCCOCCOCCOC',
            'Urea': 'NC(=O)N'
            }

SOLVENTS_REVERSE = {value: key for key, value in SOLVENTS.items()}
SOLVENT_SMILES = list(SOLVENTS.values())[1:]

# Atom feature sizes
ATOMIC_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 35, 53]
ATOM_FEATURES = {
    'atomic_num': ATOMIC_NUMBERS,
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

BOND_FDIM = 14

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2

ELECTRONEGATIVITIES = {'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'C': 2.55, 'B': 2.04, 'N': 3.04, 'O': 3.44,
                       'F': 3.98, 'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58,
                       'Cl': 3.16, 'Br': 2.96, 'I': 2.66}


def get_num_lone_pairs(atom: Chem.rdchem.Atom):
    symbol = atom.GetSymbol()
    if symbol == 'C' or symbol == 'H':
        return 0 - atom.GetFormalCharge()
    elif symbol == 'S' or symbol == 'O':
        return 2 - atom.GetFormalCharge()
    elif symbol == 'N' or symbol == 'P':
        return 1 - atom.GetFormalCharge()
    elif symbol == 'F' or symbol == 'Cl' or symbol == 'Br' or symbol == 'I':
        return 3 - atom.GetFormalCharge()
    else:
        return 0


def get_h_bond_donor(atom: Chem.rdchem.Atom) -> int:
    if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0:
        return 1
    elif atom.GetSymbol() == "O" and atom.GetTotalNumHs() > 0:
        return 2
    elif atom.GetSymbol() == "F" and atom.GetTotalNumHs() > 0:
        return 3
    else:
        return 0


def get_h_bond_acceptor(atom: Chem.rdchem.Atom):
    """returns if an atom is a h bond acceptor, 1 for F, N and O that have 1 or more lone pairs"""
    if atom.GetSymbol() == "N" and get_num_lone_pairs(atom) > 0:
        return 1
    elif atom.GetSymbol() == "O" and get_num_lone_pairs(atom) > 0:
        return 2
    elif atom.GetSymbol() == "F" and get_num_lone_pairs(atom) > 0:
        return 3
    else:
        return 0


def get_in_ring_size(atom: Chem.rdchem.Atom):
    """returns the size of the ring the atom is in (the smallest one if more rings)"""

    n = 0
    if atom.IsInRing():
        for i in reversed(range(1, 8)):
            if atom.IsInRingSize(i):
                n = i
    else:
        n = 0
    return n


def onek_encoding_unk(value, choices: List) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :return: A list containing the atom features.
    """
    # base features
    atomic_num = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    degree = onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
    formal_charge = [float(atom.GetFormalCharge())]
    total_Hs = [float(atom.GetTotalNumHs())]
    hybdridization = onek_encoding_unk(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    aromaticity = [1 if atom.GetIsAromatic() else 0]
    mass = [atom.GetMass() * 0.01]

    # solvation additional features
    n_lone_pairs = [float(get_num_lone_pairs(atom))]
    h_bond_donor = onek_encoding_unk(get_h_bond_donor(atom), [0, 1, 2, 3])
    h_bond_acceptor = onek_encoding_unk(get_h_bond_acceptor(atom), [0, 1, 2, 3])
    in_ring = [get_in_ring_size(atom)]
    electronegativity = ELECTRONEGATIVITIES[atom.GetSymbol()] if atom.GetSymbol() in ELECTRONEGATIVITIES else 0
    electronegativity = [electronegativity * 0.1]

    features = atomic_num + degree + formal_charge + total_Hs + hybdridization + aromaticity + mass + \
               n_lone_pairs + h_bond_donor + h_bond_acceptor + in_ring + electronegativity

    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """

    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bond_type = onek_encoding_unk(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE,
                                                           Chem.rdchem.BondType.DOUBLE,
                                                           Chem.rdchem.BondType.TRIPLE,
                                                           Chem.rdchem.BondType.AROMATIC])
        conjugation = [1 if bond.GetIsConjugated() else 0]
        stereo = onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
        in_ring = [1 if bond.IsInRing() else 0]

        fbond = bond_type + conjugation + stereo + in_ring

    return fbond


def mol_features(mol: Chem.Mol) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param mol: A RDKit mol.
    :return: A list containing the mol features.
    """
    topological_polar_surface_area = [AllChem.CalcTPSA(mol) / 100.0]
    molecular_radius = [Crippen.MolMR(mol) / 100.0]

    features = topological_polar_surface_area + molecular_radius

    return features


class MolGraph:
    def __init__(self, smiles):

        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.parity_atoms = []  # mapping from atom index to CW (+1), CCW (-1) or undefined tetra (0)
        self.edge_index = []  # list of tuples indicating presence of bonds

        # Convert smiles to mol
        if type(smiles) is str:
            mol = Chem.MolFromSmiles(smiles)
        else:
            mol = smiles

        self.smiles = Chem.MolToSmiles(mol)
        self.f_mol = mol_features(mol)
        self.n_atoms = mol.GetNumAtoms()

        # Get atom features
        for i, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        for _ in range(self.n_atoms):
            self.a2b.append([])

        if mol.GetNumBonds() == 0:  # TODO: FIX THIS
            self.edge_index.extend([(0, 0), (0, 0)])
            self.f_bonds.append(bond_features(None))
            self.f_bonds.append(bond_features(None))

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                self.edge_index.extend([(a1, a2), (a2, a1)])

                f_bond = bond_features(bond)

                self.f_bonds.append(f_bond)
                self.f_bonds.append(f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2
