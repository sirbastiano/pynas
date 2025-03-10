import random
import architecture_builder

# TODO: apply at chromosome, then _reparse_layers()

def gene_mutation(child, mutation_probability):
    """
    Apply mutation to a list of child chromosomes.

    Parameters:
    children (list): A list of child chromosomes.
    mutation_probability (float): The probability of a gene mutation.

    Returns:
    list: The mutated child chromosomes.
    """
    child = child.copy()
    
    for gene_index in range(len(child.chromosome)):
        rnd = random.random()
        if rnd <= mutation_probability:
            print("Mutation!")
            gene = child.chromosome[gene_index]
            # Mutate based on the type of gene
            if gene[0]=='L':  # Backbone layers
                child.chromosome[gene_index] = architecture_builder.generate_layer_code()
            elif gene[0]=='P': # Pooling layer gene
                child.chromosome[gene_index] = architecture_builder.generate_pooling_layer_code()
            elif gene[0]=='H': # Head gene
                break
            else:
                raise ValueError(f"Unrecognized gene type: {gene[0]}")
    child._reparse_layers()
    return child