def extract_reactions_from_route_az(tree, reactions=None, forward=True):

    if reactions is None:
        reactions = []

    if tree.get('is_reaction'):

        smiles = tree['metadata'].get('mapped_reaction_smiles')
        product, reactants = smiles.split('>>')

        if forward:
            smiles = f"{reactants}>>{product}"

        reactions.append(smiles)

    for child in tree.get('children', []):
        extract_reactions_from_route_az(child, reactions, forward=forward)

    # forward route
    reactions = list(reversed(reactions))

    return reactions


def extract_reactions_from_route_sp(tree, reactions=None, forward=True, parent_smiles=None):

    if reactions is None:
        reactions = []

    if tree.get('type') == 'reaction':
        product = parent_smiles
        reactants = '.'.join(
            child['smiles'] for child in tree.get('children', [])
            if child.get('type') == 'mol'
        )

        if forward:
            smiles = f"{reactants}>>{product}"
        else:
            smiles = f"{product}>>{reactants}"

        reactions.append(smiles)

    # track the nearest ancestor mol's smiles as the "current product"
    current_smiles = tree['smiles'] if tree.get('type') == 'mol' else parent_smiles

    for child in tree.get('children', []):
        extract_reactions_from_route_sp(child, reactions, forward=forward, parent_smiles=current_smiles)

    # forward route
    reactions = list(reversed(reactions))

    return reactions