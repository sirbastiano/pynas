import pandas as pd
import matplotlib.pyplot as plt


def plot_population_metrics(folder, num_generations=3, output_path=None):
    """
    Visualize metrics distribution across generations for a population.
    
    Parameters:
    -----------
    folder : str
        Path to the folder containing population data files
    num_generations : int, optional
        Number of generations to visualize (default: 3)
    output_path : str, optional
        Path to save the visualization image (default: {folder}/population_metrics.png)
    """
    # Get the last generation data
    gen = num_generations - 1
    
    df = pd.read_pickle(f'{folder}/df_population_{gen}.pkl')
    best = df[df['Fitness'] == df['Fitness'].max()]
    fitness, metric, fps, params = best.iloc[0][['Fitness', 'Metric', 'FPS', 'Params']]

    # Prepare lists to hold the full distribution for each population per generation
    gens = list(range(num_generations))

    fitnesses_data = []
    metrics_data = []
    fpses_data = []
    params_data = []

    for i in gens:
        df_gen = pd.read_pickle(f'{folder}/df_population_{i}.pkl')
        fitnesses_data.append(df_gen['Fitness'].values)
        metrics_data.append(df_gen['Metric'].values)
        fpses_data.append(df_gen['FPS'].values)
        params_data.append(df_gen['Params'].values)

    fig, axs = plt.subplots(2, 2, figsize=(16, 7), dpi=300, facecolor='white')

    axs[0, 0].violinplot(fitnesses_data, positions=gens, widths=0.7)
    axs[0, 0].set_title('Fitness Distribution per Generation')
    axs[0, 0].set_xlabel('Generation')
    axs[0, 0].set_ylabel('Fitness')

    axs[0, 1].violinplot(metrics_data, positions=gens, widths=0.7)
    axs[0, 1].set_title('Metric Distribution per Generation')
    axs[0, 1].set_xlabel('Generation')
    axs[0, 1].set_ylabel('Metric')

    axs[1, 0].violinplot(fpses_data, positions=gens, widths=0.7)
    axs[1, 0].set_title('FPS Distribution per Generation')
    axs[1, 0].set_xlabel('Generation')
    axs[1, 0].set_ylabel('FPS')

    axs[1, 1].violinplot(params_data, positions=gens, widths=0.7)
    axs[1, 1].set_title('Params Distribution per Generation')
    axs[1, 1].set_xlabel('Generation')
    axs[1, 1].set_ylabel('Params')

    plt.tight_layout()
    
    # Default output path if none specified
    if output_path is None:
        output_path = f'{folder}/population_metrics.png'
        
    # Save figure
    plt.savefig(output_path)
    plt.close(fig)


def plot_best_metrics(folder, num_generations=3, output_path=None):
    """
    Visualize the best metric values across generations.
    
    Parameters:
    -----------
    folder : str
        Path to the folder containing population data files
    num_generations : int, optional
        Number of generations to visualize (default: 3)
    output_path : str, optional
        Path to save the visualization image (default: {folder}/best_metrics.png)
    """
    # Prepare lists to hold the best values for each generation
    gens = list(range(num_generations))
    best_fitnesses = []
    best_metrics = []
    best_fpses = []
    best_params = []

    # Collect best values from each generation
    for i in gens:
        df_gen = pd.read_pickle(f'{folder}/df_population_{i}.pkl')
        
        # Get the row with the highest fitness
        best_row = df_gen[df_gen['Fitness'] == df_gen['Fitness'].max()].iloc[0]
        
        best_fitnesses.append(best_row['Fitness'])
        best_metrics.append(best_row['Metric'])
        best_fpses.append(best_row['FPS'])
        best_params.append(best_row['Params'])

    # Create the figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 7), dpi=300, facecolor='white')

    # Plot best fitness over generations
    axs[0, 0].plot(gens, best_fitnesses, 'o-', color='blue', markersize=8)
    axs[0, 0].set_title('Best Fitness per Generation')
    axs[0, 0].set_xlabel('Generation')
    axs[0, 0].set_ylabel('Fitness')
    axs[0, 0].grid(True)

    # Plot corresponding metric for best fitness model
    axs[0, 1].plot(gens, best_metrics, 'o-', color='green', markersize=8)
    axs[0, 1].set_title('Metric of Best Fitness Model per Generation')
    axs[0, 1].set_xlabel('Generation')
    axs[0, 1].set_ylabel('Metric')
    axs[0, 1].grid(True)

    # Plot corresponding FPS for best fitness model
    axs[1, 0].plot(gens, best_fpses, 'o-', color='red', markersize=8)
    axs[1, 0].set_title('FPS of Best Fitness Model per Generation')
    axs[1, 0].set_xlabel('Generation')
    axs[1, 0].set_ylabel('FPS')
    axs[1, 0].grid(True)

    # Plot corresponding params for best fitness model
    axs[1, 1].plot(gens, best_params, 'o-', color='purple', markersize=8)
    axs[1, 1].set_title('Params of Best Fitness Model per Generation')
    axs[1, 1].set_xlabel('Generation')
    axs[1, 1].set_ylabel('Params')
    axs[1, 1].grid(True)

    plt.tight_layout()
    
    # Default output path if none specified
    if output_path is None:
        output_path = f'{folder}/best_metrics.png'
        
    # Save figure
    plt.savefig(output_path)
    plt.close(fig)