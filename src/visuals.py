from src.imports import *


def plot_column_distribution(data, col1, col2):
    """
    Function to plot up to four different visualizations for two columns in the given dataset.

    Parameters:
    data (pd.DataFrame): The pandas DataFrame containing the dataset.
    col1 (str): The first column for plotting (e.g., 'age').
    col2 (str): The second column for plotting (binary: e.g., 'class').
    """
    # Create the figure and axes for up to four subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid for max 4 plots
    axes = axes.flatten()  # Flatten the 2x2 grid to easily access each axis

    # Set a common title for the figure
    plt.suptitle(f'{col1} vs {col2} Distribution', fontsize=18, fontweight='bold', color='navy')

    # Plot the first (box plot) - comparing col1 vs col2
    sns.boxplot(data=data, x=col2, y=col1, ax=axes[0], palette='Set2')
    axes[0].set_title(f'Box Plot: {col1} vs {col2}')

    # Plot the second (violin plot) - comparing col1 vs col2
    sns.violinplot(data=data, x=col2, y=col1, ax=axes[1], palette='Set2')
    axes[1].set_title(f'Violin Plot: {col1} vs {col2}')

    # Plot the third (histogram) - for col1 with hue based on col2
    sns.histplot(data=data, x=col1, hue=col2, ax=axes[2], kde=True, multiple='stack', palette='Set2')
    axes[2].set_title(f'Histogram: {col1} Distribution by {col2}')

    # Plot the fourth (optional) - scatter plot to see if there's any visible pattern for binary class
    sns.scatterplot(data=data, x=col1, y=col2, ax=axes[3], hue=col2, palette='Set2')
    axes[3].set_title(f'Scatter Plot: {col1} vs {col2}')

    # Adjust layout to avoid overlap and display the plots
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.9)  # Adjust to fit the suptitle
    plt.show()
