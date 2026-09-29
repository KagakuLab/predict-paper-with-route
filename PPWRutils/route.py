import math
from rdkit import Chem
from rdkit.Chem import Descriptors
from CGRtools import smiles
from CGRtools.containers import ReactionContainer
from synrbl import Balancer


def clean_int(x, tol=1e-2):
    if math.isclose(x, round(x), abs_tol=tol):
        return int(round(x))
    return x

def calc_reaction_atomic_economy(reaction_smiles):

    reactants_smi, product_smi = reaction_smiles.split('>>')

    reactants_mw = sum(Descriptors.MolWt(Chem.MolFromSmiles(smi)) for smi in reactants_smi.split('.'))
    product_mw = Descriptors.MolWt(Chem.MolFromSmiles(product_smi))

    reaction_ae = (product_mw / reactants_mw) * 100
    reaction_ae = clean_int(reaction_ae)

    return reaction_ae

def calc_route_atomic_economy(list_of_reaction_smiles):

    route_ae = 1.0
    for rxn_smi in list_of_reaction_smiles:
        route_ae *= calc_reaction_atomic_economy(rxn_smi) / 100

    route_ae = route_ae * 100
    return route_ae

def remove_reagents(list_of_reaction_smiles, verbose=False):

    res = []
    for r in list_of_reaction_smiles:

        rxn = smiles(r)

        not_changed_molecules = set(rxn.reactants).intersection(rxn.products)
        cgr = ~rxn
        center_atoms = set(cgr.center_atoms)

        new_reactants = []
        new_products = []
        new_reagents = []

        for molecule in rxn.reactants:
            if center_atoms.isdisjoint(molecule) or molecule in not_changed_molecules:
                new_reagents.append(molecule)
            else:
                new_reactants.append(molecule)

        for molecule in rxn.products:
            if center_atoms.isdisjoint(molecule) or molecule in not_changed_molecules:
                new_reagents.append(molecule)
            else:
                new_products.append(molecule)

        # Filter reagents by size
        new_reaction = ReactionContainer(new_reactants, new_products)
        new_reaction.name = rxn.name

        if new_reagents and verbose:
            print("Reagent removed")

        r_final = format(new_reaction, 'm')

        res.append(r_final)

    return res

def balance_reactants(list_of_reaction_smiles):

    synrbl = Balancer()
    results = synrbl.rebalance(list_of_reaction_smiles, output_dict=True)

    res = []
    for i in results:
        r_old, r_new = i["input_reaction"], i["reaction"]
        r_final = f"{r_new.split('>>')[0]}>>{r_old.split('>>')[1]}"
        res.append(r_final)

    return res
