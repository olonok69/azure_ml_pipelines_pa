"""
Visualization utilities for data analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging


def setup_plot_style():
    """
    Set up consistent matplotlib/seaborn plotting style.
    """
    # Set the style and context
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Set default figure size
    plt.rcParams["figure.figsize"] = (12, 8)

    # Set color palette
    sns.set_palette("viridis")

    # Set font properties
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]

    # Increase font sizes
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.titlesize"] = 20


def save_plot(plt, filename, output_dir="figures", dpi=300, formats=None):
    """
    Save a matplotlib plot to file in multiple formats.

    Args:
        plt: Matplotlib pyplot object
        filename: Base filename without extension
        output_dir: Directory to save the plot
        dpi: Resolution for raster formats
        formats: List of formats to save (default: ["png", "pdf", "svg"])
    """
    logger = logging.getLogger(__name__)

    if formats is None:
        formats = ["png", "pdf", "svg"]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save in each format
    for fmt in formats:
        output_path = os.path.join(output_dir, f"{filename}.{fmt}")
        try:
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight", format=fmt)
            logger.info(f"Saved plot to {output_path}")
        except Exception as e:
            logger.error(f"Error saving plot to {output_path}: {e}", exc_info=True)


def plot_categorical_distribution(
    df,
    column,
    title=None,
    figsize=(12, 8),
    top_n=None,
    sort_by_count=True,
    horizontal=True,
    show_percentages=True,
    save_as=None,
    output_dir="figures",
):
    """
    Plot distribution of categorical values.

    Args:
        df: DataFrame containing data
        column: Column name to plot
        title: Plot title (default: "{column} Distribution")
        figsize: Figure size as tuple (width, height)
        top_n: Only show top N categories (default: show all)
        sort_by_count: Sort by count in descending order
        horizontal: Use horizontal bars instead of vertical
        show_percentages: Show percentages next to counts
        save_as: Filename to save the plot (without extension)
        output_dir: Directory to save the plot
    """
    # Set up plotting style
    setup_plot_style()

    # Create figure
    plt.figure(figsize=figsize)

    # Get value counts
    counts = df[column].value_counts()

    # Sort if requested
    if sort_by_count:
        counts = counts.sort_values(ascending=False)

    # Take top N if specified
    if top_n is not None and top_n < len(counts):
        if sort_by_count:
            counts = counts.iloc[:top_n]
        else:
            counts = counts.nlargest(top_n)

    # Calculate percentages
    total = counts.sum()
    percentages = (counts / total * 100).round(1)

    # Create labels with percentages if requested
    if show_percentages:
        labels = [f"{count} ({pct}%)" for count, pct in zip(counts, percentages)]
    else:
        labels = counts

    # Create plot
    if horizontal:
        ax = sns.barplot(x=counts.values, y=counts.index, palette="viridis")
        # Add value labels to the right of bars
        for i, (count, pct) in enumerate(zip(counts, percentages)):
            ax.text(
                count + (total * 0.01),
                i,
                f"{count} ({pct}%)" if show_percentages else count,
                va="center",
            )
        # Set labels
        plt.xlabel("Count")
        plt.ylabel(column)
    else:
        ax = sns.barplot(x=counts.index, y=counts.values, palette="viridis")
        # Add value labels above bars
        for i, (count, pct) in enumerate(zip(counts, percentages)):
            ax.text(
                i,
                count + (total * 0.01),
                f"{count} ({pct}%)" if show_percentages else count,
                ha="center",
            )
        # Set labels
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")

    # Set title
    if title is None:
        title = f"{column} Distribution"
    plt.title(title)

    # Adjust layout
    plt.tight_layout()

    # Save if requested
    if save_as is not None:
        save_plot(plt, save_as, output_dir=output_dir)

    return plt


def plot_relationship_chord(
    source_values,
    target_values,
    source_name,
    target_name,
    title=None,
    figsize=(12, 10),
    cmap="viridis",
    save_as=None,
    output_dir="figures",
):
    """
    Plot relationships between source and target values as a chord diagram.

    Args:
        source_values: List of source values
        target_values: List of target values (same length as source_values)
        source_name: Name of the source category
        target_name: Name of the target category
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        save_as: Filename to save the plot (without extension)
        output_dir: Directory to save the plot

    Note: Requires matplotlib-chord package: pip install matplotlib-chord
    """
    try:
        import chord
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.error(
            "matplotlib-chord package not found. Install with: pip install matplotlib-chord"
        )
        print(
            "Error: matplotlib-chord package not found. Install with: pip install matplotlib-chord"
        )
        return None

    # Set up plotting style
    setup_plot_style()

    # Create figure
    plt.figure(figsize=figsize)

    # Create DataFrame from source and target values
    df = pd.DataFrame({source_name: source_values, target_name: target_values})

    # Get unique values
    source_unique = df[source_name].unique()
    target_unique = df[target_name].unique()

    # Count occurrences of each source-target pair
    matrix = np.zeros((len(source_unique), len(target_unique)))
    for s_idx, s_val in enumerate(source_unique):
        for t_idx, t_val in enumerate(target_unique):
            matrix[s_idx, t_idx] = (
                (df[source_name] == s_val) & (df[target_name] == t_val)
            ).sum()

    # Create labels
    all_labels = list(source_unique) + list(target_unique)

    # Create chord diagram
    chord_diagram = chord.Chord(matrix, all_labels, cmap=cmap)

    # Set title
    if title:
        plt.title(title, fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save if requested
    if save_as is not None:
        save_plot(plt, save_as, output_dir=output_dir)

    return plt


def plot_attendance_heatmap(
    visitor_session_matrix,
    title="Session Attendance Heatmap",
    figsize=(16, 12),
    cmap="viridis",
    save_as=None,
    output_dir="figures",
):
    """
    Plot session attendance as a heatmap.

    Args:
        visitor_session_matrix: DataFrame with visitors as rows, sessions as columns, and attendance counts as values
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        save_as: Filename to save the plot (without extension)
        output_dir: Directory to save the plot
    """
    # Set up plotting style
    setup_plot_style()

    # Create figure
    plt.figure(figsize=figsize)

    # Create heatmap
    ax = sns.heatmap(
        visitor_session_matrix,
        cmap=cmap,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Attendance"},
    )

    # Set title and labels
    plt.title(title, fontsize=16)
    plt.xlabel("Sessions", fontsize=14)
    plt.ylabel("Visitors", fontsize=14)

    # Adjust layout
    plt.tight_layout()

    # Save if requested
    if save_as is not None:
        save_plot(plt, save_as, output_dir=output_dir)

    return plt


def plot_neo4j_graph_preview(
    nodes,
    relationships,
    title="Neo4j Graph Preview",
    figsize=(14, 10),
    save_as=None,
    output_dir="figures",
):
    """
    Plot a preview of the Neo4j graph structure.

    Args:
        nodes: Dictionary of node labels to counts
        relationships: Dictionary of relationship types to counts
        title: Plot title
        figsize: Figure size
        save_as: Filename to save the plot (without extension)
        output_dir: Directory to save the plot

    Note: Requires networkx and matplotlib packages
    """
    try:
        import networkx as nx
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.error("networkx package not found. Install with: pip install networkx")
        print("Error: networkx package not found. Install with: pip install networkx")
        return None

    # Set up plotting style
    setup_plot_style()

    # Create figure
    plt.figure(figsize=figsize)

    # Create graph
    G = nx.DiGraph()

    # Add nodes
    node_sizes = {}
    max_node_count = max(nodes.values()) if nodes else 1
    for node_label, count in nodes.items():
        # Scale node size by count
        size = 1000 * (count / max_node_count) + 500
        node_sizes[node_label] = size
        G.add_node(node_label, size=size, count=count)

    # Add edges
    for rel_type, count in relationships.items():
        source, rel, target = rel_type.split("___")
        G.add_edge(source, target, type=rel, count=count)

    # Set up layout
    try:
        # Try spring layout first
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    except:
        # Fall back to circular layout if spring layout fails
        pos = nx.circular_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=[G.nodes[node].get("size", 1000) for node in G.nodes],
        node_color=list(range(len(G.nodes))),
        cmap="viridis",
        alpha=0.8,
    )

    # Draw edges
    edge_widths = [G.edges[edge].get("count", 1) / 100 + 1 for edge in G.edges]
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrowsize=15, alpha=0.6)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # Draw edge labels
    edge_labels = {edge: G.edges[edge].get("type", "") for edge in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # Add node count annotations
    for node, (x, y) in pos.items():
        count = G.nodes[node].get("count", 0)
        plt.text(x, y - 0.1, f"({count})", fontsize=10, ha="center")

    # Set title
    plt.title(title, fontsize=16)

    # No axis
    plt.axis("off")

    # Adjust layout
    plt.tight_layout()

    # Save if requested
    if save_as is not None:
        save_plot(plt, save_as, output_dir=output_dir)

    return plt
