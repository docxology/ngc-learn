"""
Visualization utilities for Active Inference simulations.

Provides plotting functions for free energy trajectories, beliefs, and metrics.
"""

from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import jax.numpy as jnp


def plot_free_energy(
    free_energy_trajectory: List[float],
    title: str = "Free Energy Minimization",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot free energy trajectory over inference/learning.
    
    Args:
        free_energy_trajectory: List of free energy values
        title: Plot title
        save_path: Path to save figure (if None, don't save)
        show: Whether to display plot
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    ax.plot(free_energy_trajectory, linewidth=2, color='#2E86AB')
    ax.set_xlabel('Step/Epoch', fontsize=12)
    ax.set_ylabel('Free Energy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    if len(free_energy_trajectory) > 0:
        initial = free_energy_trajectory[0]
        final = free_energy_trajectory[-1]
        ax.axhline(y=final, color='r', linestyle='--', alpha=0.5, 
                   label=f'Final: {final:.4f}')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_beliefs(
    beliefs: jnp.ndarray,
    observations: Optional[jnp.ndarray] = None,
    predictions: Optional[jnp.ndarray] = None,
    title: str = "Beliefs and Observations",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot belief states, observations, and predictions.
    
    Args:
        beliefs: Belief state values
        observations: Observed data (optional)
        predictions: Predicted observations (optional)
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
        
    Returns:
        Figure object
    """
    n_plots = 1 + (observations is not None) + (predictions is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), dpi=300)
    
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot beliefs
    im1 = axes[plot_idx].imshow(beliefs.T, aspect='auto', cmap='viridis')
    axes[plot_idx].set_title('Beliefs (Hidden States)', fontweight='bold')
    axes[plot_idx].set_xlabel('Sample')
    axes[plot_idx].set_ylabel('Dimension')
    plt.colorbar(im1, ax=axes[plot_idx])
    plot_idx += 1
    
    # Plot observations
    if observations is not None:
        im2 = axes[plot_idx].imshow(observations.T, aspect='auto', cmap='plasma')
        axes[plot_idx].set_title('Observations', fontweight='bold')
        axes[plot_idx].set_xlabel('Sample')
        axes[plot_idx].set_ylabel('Dimension')
        plt.colorbar(im2, ax=axes[plot_idx])
        plot_idx += 1
    
    # Plot predictions
    if predictions is not None:
        im3 = axes[plot_idx].imshow(predictions.T, aspect='auto', cmap='plasma')
        axes[plot_idx].set_title('Predictions', fontweight='bold')
        axes[plot_idx].set_xlabel('Sample')
        axes[plot_idx].set_ylabel('Dimension')
        plt.colorbar(im3, ax=axes[plot_idx])
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, List[float]],
    title: str = "Metrics Comparison",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot multiple metrics for comparison.
    
    Args:
        metrics_dict: Dictionary mapping metric names to value lists
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for idx, (name, values) in enumerate(metrics_dict.items()):
        color = colors[idx % len(colors)]
        ax.plot(values, label=name, linewidth=2, color=color)
    
    ax.set_xlabel('Step/Epoch', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig



