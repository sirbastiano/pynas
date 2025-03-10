import random

def single_point_crossover(parents):
    """
    Performs a single-point crossover between two parent chromosomes if they have the same depth.

    Parameters:
    parents (list): The parent chromosomes as lists of genes.

    Returns:
    list: A list containing two new child chromosomes resulting from the crossover,
          or the original parents if the crossover is not performed due to different depths.
    """

    def crossover_segment(segment1, segment2):
        min_length = min(len(segment1), len(segment2))
        start = random.randint(0, min_length)
        end = random.randint(start, min_length)
        return (segment1[:start] + segment2[start:end] + segment1[end:], 
                segment2[:start] + segment1[start:end] + segment2[end:])

    blocks_0 = (len(parents[0].chromosome) - 2) // 5
    blocks_1 = (len(parents[1].chromosome) - 2) // 5

    index_mid_0 = blocks_0 * 2
    index_mid_1 = blocks_1 * 2

    encoder_0, encoder_1 = crossover_segment(parents[0].chromosome[:index_mid_0], 
                                             parents[1].chromosome[:index_mid_1])

    decoder_0, decoder_1 = crossover_segment(parents[0].chromosome[index_mid_0 + 1:-1], 
                                             parents[1].chromosome[index_mid_1 + 1:-1])

    children = parents.copy()
    children[0].chromosome = encoder_0 + [parents[0].chromosome[index_mid_0]] + decoder_0 + [parents[0].chromosome[-1]]
    children[1].chromosome = encoder_1 + [parents[1].chromosome[index_mid_1]] + decoder_1 + [parents[1].chromosome[-1]]

    return children
