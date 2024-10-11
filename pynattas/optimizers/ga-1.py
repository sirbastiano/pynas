# GA.py
import random
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
from .. import classes
from ..functions import *
import pytorch_lightning as pl
from pytorch_lightning import Trainer






def single_point_crossover(parents):
    """
    Performs a single-point crossover between two parent chromosomes if they have the same depth.

    Parameters:
    parents (list): The parent chromosomes as lists of genes.

    Returns:
    list: A list containing two new child chromosomes resulting from the crossover,
          or the original parents if the crossover is not performed due to different depths.
    """

    #print(f"Parent 0 length: {len(parents[0].chromosome)}")
    #print(f"Parent 1 length: {len(parents[1].chromosome)}")

    blocks_0 = int((len(parents[0].chromosome) - 2) / 5)  # number of encoder/decoder blocks in parent 0
    blocks_1 = int((len(parents[1].chromosome) - 2) / 5)  # number of encoder/decoder blocks in parent 1

    #print(f"blocks for encoder/decoder in 0 are: {blocks_0}")
    #print(f"blocks for encoder/decoder in 1 are: {blocks_1}")

    # Determine the length of the shorter parent chromosome
    min_block_length = min(blocks_0, blocks_1) 
    ## = 3
    #print(f"Min block length is: {min_block_length}")

    index_mid_0 = blocks_0 * 2
    mid_0_chromosome = parents[0].chromosome[index_mid_0]
    index_mid_1 = blocks_1 * 2
    mid_1_chromosome = parents[1].chromosome[index_mid_1]
    
    min_mid_length = min(index_mid_0, index_mid_1)
    length_small_chromosome = min(len(parents[0].chromosome), len(parents[1].chromosome))
    
    #print(f"min_mid_length is: {min_mid_length}")
    
    new_encoder_0 =  parents[0].chromosome[: index_mid_0]
    #print(f"new_encoder_0 is: {new_encoder_0}")
    new_encoder_1 =  parents[1].chromosome[: index_mid_1]
    #print(f"new_encoder_1 is: {new_encoder_1}")
      
    # Randomly select crossover points for encoders and decoders
    encoder_cutoff_start = random.randint(0, min_mid_length)
    #print(f"encoder_cutoff_start: {encoder_cutoff_start}")
    encoder_cutoff_end = random.randint(encoder_cutoff_start, min_mid_length)
    #print(f"encoder_cutoff_end: {encoder_cutoff_end}")   
    
    # Perform crossover for encoders
    temp_encoder_0 = new_encoder_0[:encoder_cutoff_start] + new_encoder_1[encoder_cutoff_start:encoder_cutoff_end] + new_encoder_0[encoder_cutoff_end:]
    temp_encoder_1 = new_encoder_1[:encoder_cutoff_start] + new_encoder_0[encoder_cutoff_start:encoder_cutoff_end] + new_encoder_1[encoder_cutoff_end:]    
    
    #print(f"temp_encoder_0 is: {temp_encoder_0}")
    #print(f"temp_encoder_1 is: {temp_encoder_1}")

    
    
    
    
    new_decoder_0 = parents[0].chromosome[index_mid_0 + 1 : len(parents[0].chromosome) - 1]
    #print(f"new_decoder_0 is: {new_decoder_0}")
    new_decoder_1 = parents[1].chromosome[index_mid_1 + 1 : len(parents[1].chromosome) - 1]
    #print(f"new_decoder_1 is: {new_decoder_1}") 
    
    min_decoder_length = min(len(new_decoder_0), len(new_decoder_1))
    #print(f"min_decoder_length_is: {min_decoder_length}")
       
    
    
    decoder_cutoff_start = random.randint(0, min_decoder_length)
    #print(f"decoder_cutoff_start: {decoder_cutoff_start}")
    decoder_cutoff_end = random.randint(decoder_cutoff_start, min_decoder_length) 
    #print(f"decoder_cutoff_end: {decoder_cutoff_end}")

    # Perform crossover for decoders
    temp_decoder_0 =  new_decoder_0[:decoder_cutoff_start] + new_decoder_1[decoder_cutoff_start:decoder_cutoff_end] + new_decoder_0[decoder_cutoff_end:]
    temp_decoder_1 = new_decoder_1[:decoder_cutoff_start] + new_decoder_0[decoder_cutoff_start:decoder_cutoff_end] + new_decoder_1[decoder_cutoff_end:]
    
    #print(f"temp_decoder_0 is: {temp_decoder_0}")
    #print(f"temp_decoder_1 is: {temp_decoder_1}")
    

    # Create children
    children = parents.copy()
    children[0].chromosome = temp_encoder_0 + [parents[0].chromosome[index_mid_0]] + temp_decoder_0 + [parents[0].chromosome[-1]]
    children[1].chromosome = temp_encoder_1 + [parents[1].chromosome[index_mid_1]] + temp_decoder_1 + [parents[1].chromosome[-1]]

    return children




def mutation(children, mutation_probability):
    """
    Apply mutation to a list of child chromosomes.

    Parameters:
    children (list): A list of child chromosomes.
    mutation_probability (float): The probability of a gene mutation.

    Returns:
    list: The mutated child chromosomes.
    """
    for child in children:
        for gene_index in range(len(child.chromosome)):
            rnd = random.random()
            if rnd <= mutation_probability:
                gene = child.chromosome[gene_index]
                # Mutate based on the type of gene
                if gene[0]=='L':  # Backbone layers
                    child.chromosome[gene_index] = architecture_builder.generate_layer_code()
                elif gene[0]=='P': # Pooling layer gene
                    child.chromosome[gene_index] = architecture_builder.generate_pooling_layer_code()
                elif gene[0]=='H': # Head gene
                    child.chromosome[gene_index] = architecture_builder.generate_head_code()
                elif gene[0]=='U': # Upsampling gene
                    # child.chromosome[gene_index] = architecture_builder.generate_upsampling_layer_code()
                    pass
                elif gene.startswith('S'):  # SkipConnection gene
                    # Add logic for SkipConnection mutation if needed
                    pass
                else:
                    print("Something went wrong with mutation.")
                    exit()
    return children


def remove_duplicates(population, max_layers):
    """
    Remove duplicates from the population by replacing them with unique individuals.

    Parameters:
    population (list): A list of individuals in the population.
    max_layers (int): The maximum number of layers for an individual.

    Returns:
    list: The updated population with duplicates removed.
    """
    unique_architectures = set()
    updated_population = []

    for individual in population:
        if individual.architecture not in unique_architectures:
            unique_architectures.add(individual.architecture)
            updated_population.append(individual)
        else:
            # Attempt to generate a unique individual up to 50 times
            for _ in range(50):
                new_individual = classes.Individual(max_layers=max_layers)
                if new_individual.architecture not in unique_architectures:
                    unique_architectures.add(new_individual.architecture)
                    updated_population.append(new_individual)
                    break

    return updated_population


def ga_optimizer(max_layers, max_iter, n_individuals, mating_pool_cutoff, mutation_probability, logs_directory):
    """
    Genetic Algorithm optimizer for architecture search.

    Parameters:
    max_layers (int): The maximum number of convolutional modules for a generated individual.
    max_iter (int): The maximum number of iterations/generations.
    n_individuals (int): The number of individuals in each generation.
    mating_pool_cutoff (float): The percentage of the population used for mating.
    mutation_probability (float): The probability of gene mutation.
    logs_directory (str): The directory for log and plot files.

    Returns:
    dict: A dictionary containing the best architecture and its fitness.
    """

    # Sanity checks
    if max_layers < 1:
        print("Error: Max layers should be bigger than 0.")
        exit()
    elif n_individuals % 2 == 1:
        print("ERROR: population_size should be an even number.")
        exit()
    elif mating_pool_cutoff > 1.0:
        print("ERROR: mating_pool_cutoff should be less than 1.")
        exit()
    elif mutation_probability > 1.0:
        print("ERROR: mutation_probability should be less than 1.")
        exit()

    # Logging initialization
    mean_fitness_vector = np.zeros(shape=(max_iter + 1))
    median_fitness_vector = np.zeros_like(mean_fitness_vector)
    best_fitness_vector = np.zeros_like(mean_fitness_vector)
    iou_vector = np.zeros_like(mean_fitness_vector) ##
    fps_vector = np.zeros_like(mean_fitness_vector) ##
    model_size_vector = np.zeros_like(mean_fitness_vector) ##
                      
    # Population Initialization
    population = []
    for i in range(n_individuals):
        temp_individual = classes.Individual(max_layers=max_layers)
        population.append(temp_individual)

    population = remove_duplicates(population=population, max_layers=max_layers)

    print("Starting chromosome pool:")
    for i in population:
        parsed_layers = architecture_builder.parse_architecture_code(i.architecture)
        print(f"Architecture: {i.architecture}")
        print(f"Chromosome: {i.chromosome}")
        print(f"Inside the for loop of ga_optimizer the value of parsed_layers is: {parsed_layers}")
        #i.fitness = fitness.compute_fitness_value(parsed_layers=parsed_layers)
        #print(f"chromosome: {i.chromosome}, fitness: {i.fitness}\n")
        i.fitness, i.iou, i.fps, i.model_size = fitness.compute_fitness_value(parsed_layers=parsed_layers) ##
        print(f"chromosome: {i.chromosome}, fitness: {i.fitness}, IoU: {i.iou}, FPS: {i.fps}, Model Size: {i.model_size}\n") ##
        

    # Starting population update
    population = sorted(population, key=lambda temp: temp.fitness, reverse=True)
    historical_best_fitness = population[0].fitness
    fittest_individual = population[0].architecture
    fittest_genes = population[0].chromosome
    mean_fitness_vector[0], median_fitness_vector[0] = utils.calculate_statistics(population, attribute='fitness')
    best_fitness_vector[0] = historical_best_fitness
    iou_vector[0] = i.iou ##
    fps_vector[0] = i.fps ##
    model_size_vector[0] = i.model_size ##

    utils.show_population(
        population=population,
        generation=0,
        logs_dir=logs_directory,
        historical_best_fitness=historical_best_fitness,
        fittest_individual=fittest_individual 
    )

    # Iterations
    t = 1
    while t <= max_iter:
        print("\n" * 20)
        print(f"*** GENERATION {t} ***")
        new_population = []

        # Create a mating pool
        mating_pool = population[:int(np.floor(mating_pool_cutoff * len(population)))].copy()
        for i in range(int(np.ceil((1 - mating_pool_cutoff) * len(population)))):
            temp_individual = classes.Individual(max_layers=max_layers)
            mating_pool.append(temp_individual)

        # Coupling and mating
        couple_i = 0
        while couple_i < len(mating_pool):
            parents = [mating_pool[couple_i], mating_pool[couple_i + 1]]
            children = single_point_crossover(parents=parents)
            children = mutation(
                children=children,
                mutation_probability=mutation_probability,
            )
            new_population = new_population + children
            couple_i += 2

        # Update the population
        population = new_population.copy()
        for i in population:
            i.architecture = i.chromosome2architecture(i.chromosome)
        population = remove_duplicates(population=population, max_layers=max_layers)

        for i in population:
            parsed_layers = architecture_builder.parse_architecture_code(i.architecture)
            print(f"Architecture: {i.architecture}")
            print(f"Chromosome: {i.chromosome}")
            #i.fitness = fitness.compute_fitness_value(parsed_layers=parsed_layers)
            #print(f"chromosome: {i.chromosome}, fitness: {i.fitness}\n")
            i.fitness, i.iou, i.fps, i.model_size = fitness.compute_fitness_value(parsed_layers=parsed_layers) ##
            print(f"chromosome: {i.chromosome}, fitness: {i.fitness}, IoU: {i.iou}, FPS: {i.fps}, Model Size: {i.model_size}\n") ##


        # Update historical best
        population = sorted(population, key=lambda temp: temp.fitness, reverse=True)
        if historical_best_fitness <= population[0].fitness:
            historical_best_fitness = population[0].fitness
            fittest_individual = population[0].architecture
            fittest_genes = population[0].chromosome

        if t == max_iter:
            print(f"THE LAST GENERATION ({t}):")
        print(f"For generation {t}, the best fitness of the population is {population[0].fitness}.")
        print(f"The best historical fitness is {historical_best_fitness},"
              f"with the most fit individual having the following genes: {fittest_genes}.")

        utils.show_population(
            population=population,
            generation=t,
            logs_dir=logs_directory,
            historical_best_fitness=historical_best_fitness,
            fittest_individual=fittest_individual,
        )

        # Update analytics
        mean_fitness_vector[t], median_fitness_vector[t] = utils.calculate_statistics(population, attribute='fitness')
        best_fitness_vector[t] = historical_best_fitness
        iou_vector[t] = population[0].iou  # Assuming you store IoU as an attribute
        fps_vector[t] = population[0].fps  # Assuming you store FPS as an attribute
        model_size_vector[t] = population[0].model_size  # Assuming you store model size as an attribute


        t += 1

    plt.figure(figsize=(12, 8))

    iterations = np.arange(0, max_iter + 1, 1)
    plt.plot(iterations, mean_fitness_vector, color='red', label='mean')
    plt.plot(iterations, median_fitness_vector, color='blue', label='median')
    plt.plot(iterations, best_fitness_vector, color='green', label='best')

    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.grid()
    plt.title("Mean, Median, and Historical Best fitness over the iterations")
    plt.legend()
    plt.savefig(os.path.join(logs_directory, f'plot_stats_over_iterations.png'), bbox_inches='tight')
    # plt.show()
    
    plt.figure(figsize=(12, 8))
    plt.plot(iterations, iou_vector, color='orange', label='IoU')
    plt.xlabel("Iteration")
    plt.ylabel("IoU")
    plt.grid()
    plt.title("IoU over the iterations")
    plt.legend()
    plt.savefig(os.path.join(logs_directory, 'plot_iou_over_iterations.png'), bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    plt.plot(iterations, fps_vector, color='purple', label='FPS')
    plt.xlabel("Iteration")
    plt.ylabel("FPS")
    plt.grid()
    plt.title("FPS over the iterations")
    plt.legend()
    plt.savefig(os.path.join(logs_directory, 'plot_fps_over_iterations.png'), bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    plt.plot(iterations, model_size_vector, color='cyan', label='Model Size')
    plt.xlabel("Iteration")
    plt.ylabel("Model Size (parameters)")
    plt.grid()
    plt.title("Model Size over the iterations")
    plt.legend()
    plt.savefig(os.path.join(logs_directory, 'plot_model_size_over_iterations.png'), bbox_inches='tight')


    # The final result is the position of the alpha
    best_fit = {
        "position": fittest_individual,
        "fitness": historical_best_fitness,
    }

    return best_fit
