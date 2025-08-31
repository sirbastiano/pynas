import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple, Any

# --- Configuration ---
# IMPORTANT: Adjust these paths if your environment differs from the original script's environment.
PICKLE_DIR = './data'  # Directory where the pickle files are stored
OUTPUT_FILENAME = 'nas_evolution_plot.pdf'  # Output filename for the plot
NUM_GENERATIONS_TO_PROCESS = 15  # Number of generations to process

# --- Helper Annotation Functions ---
def add_annotation(ax: plt.Axes, x: float, y: float, text: str, color: str, offset_x: float = -2, offset_y: float = 1.1) -> None:
    """Adds an annotation to the plot with an arrow."""
    ax.annotate(f'{text}',
                xy=(x, y),
                xytext=(x + offset_x, y * offset_y),
                arrowprops=dict(facecolor=color, shrink=0.05, width=0.5, headwidth=3, headlength=3),
                fontsize=7, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor=color))

def add_annotation_1(ax: plt.Axes, x: float, y: float, text: str, color: str, dx: float = -10, dy: float = 10) -> None:
    """Adds an annotation to the plot with an arrow, using offset points."""
    ax.annotate(f'{text}',
                xy=(x, y),
                xytext=(dx, dy),
                textcoords='offset points',
                arrowprops=dict(facecolor=color, shrink=0.05, width=0.5, headwidth=3, headlength=3),
                fontsize=7, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor=color))

# --- Data Loading and Preprocessing Functions ---
def load_pickle_data(pickle_dir_path: str) -> Dict[int, pd.DataFrame]:
    """
    Loads all df_population_*.pkl files from the specified directory.

    Args:
        pickle_dir_path: Path to the directory containing pickle files.

    Returns:
        A dictionary where keys are generation numbers and values are pandas DataFrames.
    """
    df_population_files: List[str] = glob.glob(os.path.join(pickle_dir_path, 'df_population_*.pkl'))
    all_dataframes: Dict[int, pd.DataFrame] = {}
    if not df_population_files:
        print(f"Warning: No pickle files found matching 'df_population_*.pkl' in {pickle_dir_path}")
        return all_dataframes

    for file_path in df_population_files:
        try:
            gen_number_str = os.path.basename(file_path).split('_')[-1].split('.')[0]
            if not gen_number_str.isdigit():
                print(f"Warning: Could not parse generation number from filename {file_path}. Skipping.")
                continue
            gen_number = int(gen_number_str)
            
            # Try loading directly with pd.read_pickle first
            data: Any = pd.read_pickle(file_path)
            
            if isinstance(data, pd.DataFrame):
                all_dataframes[gen_number] = data
            elif isinstance(data, list) and data and isinstance(data[0], pd.DataFrame):
                # This case might be less common if pd.read_pickle succeeds directly with a DataFrame
                all_dataframes[gen_number] = data[0]
            elif isinstance(data, dict) and data and isinstance(next(iter(data.values()), None), pd.DataFrame):
                # This case might also be less common
                all_dataframes[gen_number] = next(iter(data.values()))
            else:
                print(f"Warning: Successfully read pickle from {file_path} but it was not a DataFrame or known structure. Data type: {type(data)}.")

        except TypeError as te:
            if "Argument 'placement' has incorrect type" in str(te):
                print(f"Error loading or processing {file_path} with pd.read_pickle: {te}")
                print(f"  This error strongly suggests a pandas version incompatibility.")
                print(f"  The pickle file was likely created with a different version of pandas.")
                print(f"  Consider using the pandas version that created the pickle, or re-generating the pickles.")
                print(f"  The custom loading logic in cell 7 of your notebook ('run_print_v2_analysis.ipynb') might offer an alternative.")
            else:
                print(f"TypeError loading or processing {file_path} with pd.read_pickle: {te}")
        except Exception as e:
            # Fallback to generic pickle.load if pd.read_pickle fails for other reasons (e.g., not a pandas pickle)
            # This part is now more of a fallback or for non-DataFrame pickles if those are expected.
            # Given the error, the primary issue is with pandas DataFrame unpickling.
            print(f"pd.read_pickle failed for {file_path}: {e}. Attempting generic pickle.load...")
            try:
                with open(file_path, 'rb') as f:
                    generic_data: Any = pickle.load(f)
                if isinstance(generic_data, pd.DataFrame):
                    all_dataframes[gen_number] = generic_data
                elif isinstance(generic_data, list) and generic_data and isinstance(generic_data[0], pd.DataFrame):
                    all_dataframes[gen_number] = generic_data[0]
                elif isinstance(generic_data, dict) and generic_data and isinstance(next(iter(generic_data.values()), None), pd.DataFrame):
                    all_dataframes[gen_number] = next(iter(generic_data.values()))
                else:
                    print(f"Warning: Could not extract DataFrame from {file_path} using generic pickle.load. Data type: {type(generic_data)}.")
            except Exception as e2:
                print(f"Generic pickle.load also failed for {file_path}: {e2}")
                
    return all_dataframes

def calculate_evolution_metrics(all_dataframes: Dict[int, pd.DataFrame], num_generations_limit: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Calculates evolution metrics across generations.

    Args:
        all_dataframes: Dictionary of DataFrames per generation.
        num_generations_limit: The number of earliest generations to process.

    Returns:
        A tuple containing:
            - evolution_df: DataFrame with aggregated metrics per generation.
            - fitness_gap: Series representing the fitness gap.
    """
    if not all_dataframes:
        return pd.DataFrame(columns=['Generation', 'Max Fitness', 'Max Metric', 'Max FPS', 'Min Params', 'Mean Fitness', 'Min Fitness']), pd.Series(dtype=float)

    # Process the earliest 'num_generations_limit' generations available
    sorted_gen_keys = sorted(all_dataframes.keys())
    generations_to_process = sorted_gen_keys[:num_generations_limit]

    evolution_data: List[Dict[str, Any]] = []
    for gen in generations_to_process:
        if gen not in all_dataframes:
            print(f"Warning: Generation {gen} (selected for processing) not found in loaded data.")
            continue
        df_gen = all_dataframes[gen]
        if 'Fitness' not in df_gen.columns:
            print(f"Warning: 'Fitness' column missing in data for generation {gen}.")
            continue

        max_fit = df_gen['Fitness'].max()
        row_data = {'Metric': np.nan, 'FPS': np.nan, 'Params': np.nan}

        if not pd.isna(max_fit):
            fittest_rows = df_gen[df_gen['Fitness'] == max_fit]
            if not fittest_rows.empty:
                row = fittest_rows.iloc[0]
                row_data = {
                    'Metric': row.get('Metric', np.nan),
                    'FPS': row.get('FPS', np.nan),
                    'Params': row.get('Params', np.nan) # This is 'Params' of the fittest model
                }
        
        evolution_data.append({
            'Generation': gen,
            'Max Fitness': max_fit,
            'Max Metric': row_data['Metric'],
            'Max FPS': row_data['FPS'],
            'Min Params': row_data['Params'], 
            'Mean Fitness': df_gen['Fitness'].mean()
        })

    evolution_df = pd.DataFrame(evolution_data)
    if evolution_df.empty:
        return pd.DataFrame(columns=['Generation', 'Max Fitness', 'Max Metric', 'Max FPS', 'Min Params', 'Mean Fitness', 'Min Fitness']), pd.Series(dtype=float)

    evolution_df = evolution_df.sort_values('Generation')

    min_fitness_values: List[float] = []
    for gen_val in evolution_df['Generation']: # Use gen_val to avoid conflict with outer scope 'gen' if any
        if gen_val in all_dataframes and 'Fitness' in all_dataframes[gen_val].columns:
            gen_min = all_dataframes[gen_val]['Fitness'].replace(0, float('nan')).min()
            min_fitness_values.append(gen_min)
        else:
            min_fitness_values.append(np.nan)
    evolution_df['Min Fitness'] = min_fitness_values

    fitness_gap = pd.Series(dtype=float)
    if not evolution_df.empty and 'Max Fitness' in evolution_df.columns and not evolution_df['Max Fitness'].empty:
        last_gen_max_fitness = evolution_df['Max Fitness'].iloc[-1]
        fitness_gap = evolution_df['Max Fitness'] - last_gen_max_fitness
        
    return evolution_df, fitness_gap

def get_top_models(all_dataframes: Dict[int, pd.DataFrame], top_n: int = 20) -> pd.DataFrame:
    """
    Combines models from all generations, removes duplicates, and returns the top N models by fitness.

    Args:
        all_dataframes: Dictionary of DataFrames per generation.
        top_n: Number of top models to return.

    Returns:
        DataFrame of the top N models.
    """
    if not all_dataframes:
        return pd.DataFrame()

    all_models_list: List[pd.DataFrame] = []
    for gen, df in all_dataframes.items():
        df_copy = df.copy()
        df_copy['Generation'] = gen
        all_models_list.append(df_copy)
    
    if not all_models_list:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_models_list, ignore_index=True)

    for col in combined_df.columns:
        # Check if any value in the column is a list or dict
        is_unhashable = False
        try:
            # Efficient check for the first few non-NaN values
            for x in combined_df[col].dropna().head(): 
                if isinstance(x, (list, dict)):
                    is_unhashable = True
                    break
            if is_unhashable or combined_df[col].apply(type).isin([list, dict]).any(): # Fallback to slower check if needed
                 print(f"Converting unhashable column '{col}' to string for duplicate removal.")
                 combined_df[col] = combined_df[col].astype(str)
        except Exception as e: # Broad exception for safety during type checking / conversion
            print(f"Could not process column {col} for unhashable check/conversion: {e}. It might cause issues in drop_duplicates.")

    subset_cols = [col for col in ['Fitness', 'Metric', 'FPS', 'Params'] if col in combined_df.columns]
    if not subset_cols:
        print("Warning: Key columns (Fitness, Metric, FPS, Params) for duplicate check not found. Duplicates might persist.")
        combined_df_no_duplicates = combined_df
    else:
        try:
            combined_df_no_duplicates = combined_df.drop_duplicates(subset=subset_cols)
        except TypeError as e:
            print(f"Error during drop_duplicates (possibly due to remaining unhashable types): {e}. Returning combined_df without duplicate removal by subset.")
            combined_df_no_duplicates = combined_df

    if 'Fitness' not in combined_df_no_duplicates.columns:
        print("Warning: 'Fitness' column not in DataFrame. Cannot select top N models.")
        return pd.DataFrame()

    return combined_df_no_duplicates.nlargest(top_n, 'Fitness')

# --- Plotting Configuration and Functions ---
def setup_plot_style(k=1.4) -> None:
    """Sets the global matplotlib plot style with serif fonts."""
    plt.style.use('seaborn-v0_8-whitegrid')  # Colorblind-friendly style with clear contrast
    # Other good options: 'ggplot', 'fivethirtyeight', 'bmh', 'seaborn-v0_8-dark'
    #Other good options: 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-whitegrid'
    # Set gridline style to dashed
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.6
    
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'mathtext.fontset': 'dejavusans',
        'font.size': int(7*k),
        'axes.titlesize': int(7*k),
        'axes.labelsize': int(7*k),
        'xtick.labelsize': int(7*k),
        'ytick.labelsize': int(7*k),
        'legend.fontsize': int(6*k),
        'figure.titlesize': int(12*k)
    })


# --- Plotting Functions 1 ---
def plot_sota_comparison(ax: plt.Axes, sota_models_data: Dict[str, List[Any]], colors: Dict[str, str], sizes: List[float]) -> None:
    """
    Plots the State-of-the-Art (SOTA) model comparison.
    Args:
        ax: Matplotlib axis object.
        sota_models_data: Dict with 'models', 'accuracies' (Metric), 'latencies' (FPS).
        colors: Dictionary mapping model names to colors.
        sizes: List of sizes for scatter plot points (related to num_params).
    """
    models = sota_models_data['models']
    accuracies = sota_models_data['accuracies'] # Metric
    fps_values = sota_models_data['latencies']  # Assumed to be FPS

    for i, model_name in enumerate(models):
        ax.scatter(fps_values[i], accuracies[i], s=sizes[i], color=colors.get(model_name, 'grey'), alpha=0.7)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=model_name,
                              markerfacecolor=colors.get(model_name, 'grey'), markersize=6) for model_name in models]
    ax.legend(handles=legend_elements, fontsize=6, 
              loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=len(models)/2, frameon=True)

    if "Best PyNAS" in models:
        best_index = models.index("Best PyNAS")
        ax.annotate("Best PyNAS",
                    xy=(fps_values[best_index], accuracies[best_index]),
                    xytext=(fps_values[best_index] - 40., accuracies[best_index] - 0.05),
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="black", lw=1),
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    ax.set_xlabel("FPS")
    ax.set_ylabel("Metric")
    ax.set_xlim(0, 200) # ORIN
    ax.set_ylim(0.75, 0.95)
   
    # Myriad data:
    # ax.set_ylim(0.6, 0.82)
    # ax.set_xlim(0., 15)
    ax.text(0.1, 0.915, "(a)", fontsize=12, ha='center', va='bottom', transform=ax.transAxes)




# --- Plotting Functions 2 ---
def plot_pareto_front(ax: plt.Axes, top_models_df: pd.DataFrame, ylim=[160, 185]) -> None:
    """
    Plots the Pareto front of the top models.
    Args:
        ax: Matplotlib axis object.
        top_models_df: DataFrame of top models ('FPS', 'Metric', 'Generation' required).
    """
    required_cols = ['FPS', 'Metric', 'Generation']
    if top_models_df.empty or not all(col in top_models_df.columns for col in required_cols):
        print(f"Warning: top_models_df is empty or missing one of {required_cols} for Pareto plot.")
        ax.text(0.5, 0.5, "Data unavailable", ha='center', va='center', transform=ax.transAxes)
    else:
        scatter = ax.scatter(top_models_df['Metric'], top_models_df['FPS'], alpha=0.8,
                   c=top_models_df['Generation'], cmap='viridis', s=30, edgecolor='k', linewidth=0.3)
        best_model = top_models_df.iloc[0]  # Assumes df is sorted by fitness (nlargest)
        ax.scatter([best_model['Metric']], [best_model['FPS']], s=70, facecolors='none', edgecolors='r', linewidth=1.5)
        add_annotation(ax, best_model['Metric'], best_model['FPS'], 
                       f"Best PyNAS:\n{best_model['FPS']:.0f} FPS", 'red', -0.021, 1.02)
        
        # Add horizontal colorbar at the top
        cax = ax.inset_axes([0, 1.05, 1, 0.05])  # [x, y, width, height] in axes coordinates
        cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')
        cbar.set_label('Generation', fontsize=7, labelpad=3)
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

    ax.set_xlabel('Metric')
    ax.set_ylabel('FPS')
    
    ax.text(0.1, 0.915, "(a)", fontsize=12, ha='center', va='bottom', transform=ax.transAxes)
    ax.set_ylim(ylim[0], ylim[1])  # Consistent with original
    # ax.grid(True, alpha=0.3, linestyle='--')



# --- Plotting Functions 3 ---
def plot_fitness_vs_generation(ax: plt.Axes, evolution_df: pd.DataFrame, fitness_gap: pd.Series) -> None:
    """
    Plots fitness metrics (Max, Min, Gap) and Parameters vs. Generation.
    Args:
        ax: Matplotlib axis object for fitness.
        evolution_df: DataFrame with evolution metrics.
        fitness_gap: Series for fitness gap.
    """
    required_cols = ['Generation', 'Max Fitness', 'Min Fitness', 'Min Params']
    if evolution_df.empty or not all(col in evolution_df.columns for col in required_cols):
        print(f"Warning: evolution_df is empty or missing one of {required_cols} for Fitness plot.")
        ax.text(0.5, 0.5, "Data unavailable", ha='center', va='center', transform=ax.transAxes)
    else:
        latest_gen = evolution_df['Generation'].iloc[-1]
        ax.plot(evolution_df['Generation'], evolution_df['Max Fitness'], 'o-', color='tab:blue', linewidth=1, markersize=2, label='Max Fitness')
        ax.plot(evolution_df['Generation'], evolution_df['Min Fitness'], 's-', color='tab:orange', linewidth=1, markersize=2, label='Min Fitness')
        if not fitness_gap.empty:
            ax.plot(evolution_df['Generation'], fitness_gap.abs(), 'd-', color='tab:purple', linewidth=1, markersize=2, label='Fitness Gap')
        ax.fill_between(evolution_df['Generation'], evolution_df['Max Fitness'], evolution_df['Min Fitness'], alpha=0.15, color='tab:blue')

        ax_twin = ax.twinx()
        ax_twin.plot(evolution_df['Generation'], evolution_df['Min Params'], '^-', color='green', linewidth=1, markersize=2, label='Parameters')
        ax_twin.set_ylabel('Parameters', color='green')
        ax_twin.set_yscale('log')
        ax_twin.tick_params(axis='y', labelcolor='green')
        ax_twin.grid(False)
        ax_twin.set_ylim(1e5, 1e8)  # Set y-limits for parameters
        ax_twin.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])  # Fixed y-ticks for parameters

        # Annotations
        add_annotation(ax, latest_gen, evolution_df['Max Fitness'].iloc[-1], f"{evolution_df['Max Fitness'].iloc[-1]:.2f}", 'tab:blue', -2.5, 1.1)
        if not pd.isna(evolution_df['Min Fitness'].iloc[-1]):
            add_annotation(ax, latest_gen, evolution_df['Min Fitness'].iloc[-1], f"{evolution_df['Min Fitness'].iloc[-1]:.2f}", 'tab:orange', -1, 0.85)
        if not fitness_gap.empty and not pd.isna(fitness_gap.iloc[-1]):
            add_annotation_1(ax, latest_gen, fitness_gap.iloc[-1], f"{fitness_gap.iloc[-1]:.2f}", 'tab:purple', dx=-25, dy=15)
        if not pd.isna(evolution_df['Min Params'].iloc[-1]):
            add_annotation(ax_twin, latest_gen, evolution_df['Min Params'].iloc[-1], f"{evolution_df['Min Params'].iloc[-1]/1000:.1f}K", 'green', -.75, 2.5)
        
        # Add legends for both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=6, 
                  bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=True)
            
    ax.set_xlabel('Generation')
    ax.set_xlim(-0.5, 9.5) # Consistent with original
    ax.set_ylim(-0.5, 2) # Consistent with original
    ax.set_ylabel('Fitness / Gap')
    ax.text(0.1, 0.915, "(b)", fontsize=12, ha='center', va='bottom', transform=ax.transAxes)
    # ax.grid(True, alpha=0.3, linestyle='--')




# --- Plotting Functions 4 ---
def plot_metric_fps_vs_generation(ax: plt.Axes, evolution_df: pd.DataFrame) -> None:
    """
    Plots Max Metric and Max FPS vs. Generation.
    Args:
        ax: Matplotlib axis object for Metric.
        evolution_df: DataFrame with evolution metrics.
    """
    required_cols = ['Generation', 'Max Metric', 'Max FPS']
    if evolution_df.empty or not all(col in evolution_df.columns for col in required_cols):
        print(f"Warning: evolution_df is empty or missing one of {required_cols} for Metric/FPS plot.")
        ax.text(0.5, 0.5, "Data unavailable", ha='center', va='center', transform=ax.transAxes)
    else:
        latest_gen = evolution_df['Generation'].iloc[-1]
        ax.plot(evolution_df['Generation'], evolution_df['Max Metric'], 'o-', color='tab:green', linewidth=1, markersize=2, label='Max Metric')
        ax.tick_params(axis='y', labelcolor='tab:green')

        ax_twin = ax.twinx()
        ax_twin.plot(evolution_df['Generation'], evolution_df['Max FPS'], 's-', color='tab:red', linewidth=1, markersize=2, label='Max FPS')
        ax_twin.set_ylabel('FPS', color='tab:red')
        ax_twin.tick_params(axis='y', labelcolor='tab:red')
        ax_twin.grid(False)

        # Annotations
        if not pd.isna(evolution_df['Max Metric'].iloc[-1]):
            add_annotation(ax, latest_gen, evolution_df['Max Metric'].iloc[-1], f"{evolution_df['Max Metric'].iloc[-1]:.3f}", 'tab:green', -1, 1.1)
        if not pd.isna(evolution_df['Max FPS'].iloc[-1]):
            add_annotation(ax_twin, latest_gen, evolution_df['Max FPS'].iloc[-1], f"{evolution_df['Max FPS'].iloc[-1]:.1f}", 'tab:red', -2, 0.999)

        # Add legends for both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=6, 
                 bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=True)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Metric', color='tab:green')
    # set ylim to match the original script
    # ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yticks(np.arange(0.7, 1.1, 0.1)) # Consistent with original
    ax.text(0.1, 0.915, "(c)", fontsize=12, ha='center', va='bottom', transform=ax.transAxes)
    ax.set_xlim(-.5, 9.5) # Consistent with original




# --- Animation Function ---
def create_evolution_animation(all_dataframes: Dict[int, pd.DataFrame], 
                                output_filename: str = 'evolution_animation.gif', 
                                interval: int = 500) -> None:
    """
    Creates an animated GIF showing the evolution of fitness, metric, and FPS over generations.
    
    Args:
        all_dataframes: Dictionary of DataFrames per generation.
        output_filename: Name of the output GIF file.
        interval: Time between frames in milliseconds.
    """
    import matplotlib.animation as animation
    
    setup_plot_style(k=1.0)  # Use smaller font size for animation
    
    # Get sorted generations
    sorted_gen_keys = sorted(all_dataframes.keys())
    if not sorted_gen_keys:
        print("No generations found. Cannot create animation.")
        return
    
    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=300)
    
    # Set up twin axes for the second y-axis on each plot
    ax1_twin = ax1.twinx()
    ax2_twin = ax2.twinx()
    
    # Initial empty data
    max_fitness_data = []
    min_fitness_data = []
    fitness_gap_data = []
    min_params_data = []
    max_metric_data = []
    max_fps_data = []
    x_data = []
    
    # Initial plots
    line_max_fitness, = ax1.plot([], [], 'o-', color='tab:blue', linewidth=1, markersize=2, label='Max Fitness')
    line_min_fitness, = ax1.plot([], [], 's-', color='tab:orange', linewidth=1, markersize=2, label='Min Fitness')
    line_fitness_gap, = ax1.plot([], [], 'd-', color='tab:purple', linewidth=1, markersize=2, label='Fitness Gap')
    fill_between = ax1.fill_between([], [], [], alpha=0.15, color='tab:blue')
    line_min_params, = ax1_twin.plot([], [], '^-', color='green', linewidth=1, markersize=2, label='Parameters')
    
    line_max_metric, = ax2.plot([], [], 'o-', color='tab:green', linewidth=1, markersize=2, label='Max Metric')
    line_max_fps, = ax2_twin.plot([], [], 's-', color='tab:red', linewidth=1, markersize=2, label='Max FPS')
    
    # Set titles and labels
    ax1.set_title('Fitness Evolution')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness / Gap')
    ax1_twin.set_ylabel('Parameters', color='green')
    ax1_twin.set_yscale('log')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    ax1_twin.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6])  # Fixed y-ticks for parameters
    
    ax2.set_title('Metric and FPS Evolution')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Metric', color='tab:green')
    ax2_twin.set_ylabel('FPS', color='tab:red')
    ax2_twin.tick_params(axis='y', labelcolor='tab:red')
    
    # Set initial consistent limits similar to the main plots
    ax1.set_xlim(-0.5, 15.5)
    ax2.set_xlim(-0.5, 15.5)
    ax2.set_yticks(np.arange(0.7, 1.1, 0.1))
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=6)
    
    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, loc='lower left', fontsize=6)
    
    # Function to update the plots for each frame
    def update(frame):
        # Ensure frame is within valid range of available generations
        if not sorted_gen_keys:
            return line_max_fitness, line_min_fitness, line_fitness_gap, fill_between, line_min_params, line_max_metric, line_max_fps
        
        # Process up to the current generation
        frame_idx = min(frame, len(sorted_gen_keys)-1)
        current_gen = sorted_gen_keys[frame_idx]
        current_dataframes = {gen: df for gen, df in all_dataframes.items() if gen <= current_gen}
        
        # Calculate metrics for the current set of generations
        evolution_df, fitness_gap = calculate_evolution_metrics(current_dataframes, NUM_GENERATIONS_TO_PROCESS)
        
        if evolution_df.empty:
            return line_max_fitness, line_min_fitness, line_fitness_gap, fill_between, line_min_params, line_max_metric, line_max_fps
        
        # Update data
        x_data = evolution_df['Generation'].values
        max_fitness_data = evolution_df['Max Fitness'].values
        min_fitness_data = evolution_df['Min Fitness'].values
        fitness_gap_data = fitness_gap.abs().values if not fitness_gap.empty else []
        min_params_data = evolution_df['Min Params'].values
        max_metric_data = evolution_df['Max Metric'].values
        max_fps_data = evolution_df['Max FPS'].values
        
        # Update plots
        line_max_fitness.set_data(x_data, max_fitness_data)
        line_min_fitness.set_data(x_data, min_fitness_data)
        
        if len(fitness_gap_data) > 0:
            line_fitness_gap.set_data(x_data, fitness_gap_data)
        
        # Update fill between (need to clear and recreate)
        for coll in ax1.collections[:]:
            coll.remove()
        fill_between = ax1.fill_between(x_data, max_fitness_data, min_fitness_data, alpha=0.15, color='tab:blue')
        
        line_min_params.set_data(x_data, min_params_data)
        line_max_metric.set_data(x_data, max_metric_data)
        line_max_fps.set_data(x_data, max_fps_data)
        
        # Keep consistent x limits for better visualization
        ax1.set_xlim(-0.5, 15.5)
        ax2.set_xlim(-0.5, 15.5)
        
        # Only update y limits if data exceeds current bounds
        if min(min_fitness_data) < ax1.get_ylim()[0] or max(max_fitness_data) > ax1.get_ylim()[1]:
            y_min = min(min_fitness_data) * 0.95
            y_max = max(max_fitness_data) * 1.05
            if len(fitness_gap_data) > 0:
                y_min = min(y_min, min(fitness_gap_data) * 0.95)
                y_max = max(y_max, max(fitness_gap_data) * 1.05)
            ax1.set_ylim(y_min, y_max)
        
        # Parameters axis always in log scale with fixed range
        ax1_twin.set_ylim(1e2, 1e6)
        
        # Metric axis with consistent range
        if ax2.get_ylim()[0] > min(max_metric_data) * 0.95 or ax2.get_ylim()[1] < max(max_metric_data) * 1.05:
            ax2.set_ylim(0.7, 1.0)  # Use a consistent range
        
        # FPS axis with consistent range 
        if min(max_fps_data) < ax2_twin.get_ylim()[0] or max(max_fps_data) > ax2_twin.get_ylim()[1]:
            fps_min = max(0, min(max_fps_data) * 0.9)  # Ensure positive
            fps_max = max(max_fps_data) * 1.1
            ax2_twin.set_ylim(fps_min, fps_max)
        
        # Add generation number in the plot
        fig.suptitle(f'Generation: {current_gen}', fontsize=12)
        
        return line_max_fitness, line_min_fitness, line_fitness_gap, fill_between, line_min_params, line_max_metric, line_max_fps
    
    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(sorted_gen_keys), 
        blit=False, interval=interval, repeat=True
    )
    
    # Save as GIF with higher DPI
    ani.save(output_filename, writer='pillow', fps=2, dpi=400)
    print(f"Animation saved to {output_filename}")
    
    plt.close(fig)


# --- Main Execution ---
def main() -> None:
    """
    Main function to load data, process it, and generate plots.
    """
    setup_plot_style()

    all_dataframes = load_pickle_data(PICKLE_DIR)
    if not all_dataframes:
        print(f"No data loaded from {PICKLE_DIR}. Ensure PICKLE_DIR is correct and files exist. Exiting.")
        return

    evolution_df, fitness_gap = calculate_evolution_metrics(all_dataframes, NUM_GENERATIONS_TO_PROCESS)
    top20_models = get_top_models(all_dataframes, top_n=20)

    print("Evolution DataFrame:")
    print(evolution_df.to_string()) # Print full df
    print("\nTop 20 Models:")
    if not top20_models.empty:
        cols_to_print = [col for col in ['Generation', 'Fitness', 'Metric', 'FPS', 'Params'] if col in top20_models.columns]
        print(top20_models[cols_to_print].to_string())
    else:
        print("No top models to display.")

    # SOTA comparison data (MYRIAD data from original script)
    sota_data: Dict[str, List[Any]] = {
        "models": ["EfficientnetB0", "Mobileone_s0", "ResNet18", "Best PyNAS"],
        "accuracies": [0.79526335, 0.683842, 0.7708845, 0.802], # Metric
        "latencies": [4.73227, 4.32405, 5.8624787, 11.5],      # FPS
        "num_params": [6253056, 8588148, 14341188, 900]
    }
    sota_colors: Dict[str, str] = {
        "EfficientnetB0": "blue", "Mobileone_s0": "green",
        "ResNet18": "red", "Best PyNAS": "purple"
    }
    sota_sizes: List[float] = [p / 10000 for p in sota_data["num_params"]]

    fig, axs = plt.subplots(1, 4, figsize=(16, 4), dpi=400)

    plot_sota_comparison(axs[0], sota_data, sota_colors, sota_sizes)
    plot_pareto_front(axs[1], top20_models)
    plot_fitness_vs_generation(axs[2], evolution_df, fitness_gap)
    plot_metric_fps_vs_generation(axs[3], evolution_df)

    fig.tight_layout()
    try:
        fig.savefig(OUTPUT_FILENAME, format='pdf', dpi=400, bbox_inches='tight')
        print(f"Plot saved to {OUTPUT_FILENAME}")
    except Exception as e:
        print(f"Error saving plot to {OUTPUT_FILENAME}: {e}")
    plt.close(fig)

if __name__ == '__main__':
    main()
