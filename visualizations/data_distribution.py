
import numpy as np
import matplotlib.pyplot as plt


def expected_data_split(
        min_op: int = 1,
        max_op: int = 16,
        n_samples: int = 100_000,
        seed: int = 0,
    ) -> np.ndarray:
    """
    Generates expected data split.
    
    Args:
        min_op: Minimum value for the random integers.
        max_op: Maximum value for the random integers.
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Array of minimum values between two sets of random integers.
    """
    np.random.seed(seed)
    a = np.random.randint(min_op, max_op, size=n_samples)
    b = np.random.randint(min_op, max_op, size=n_samples)
    c = np.minimum(a, b)
    return c


def plot_expected_data_distribution(
        expected_data: np.ndarray,
        title: str = 'Expected Data Distribution',
        **kwargs,
    ) -> None:
    """
    Plots the distribution of expected data as histogram.

    Args:
        expected_data: Array of expected data values.
        title: Title for the plot.
    """
    plt.figure(figsize=(4.5, 3.5))
    plt.hist(
        expected_data,
        bins=np.arange(expected_data.min(), expected_data.max() + 2) - 0.5,
        width=0.9,
        density=True)
    plt.title(title)
    plt.xlabel('difficulty (op count)')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(expected_data.min(), expected_data.max() + 1))
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    expected_data = expected_data_split()
    plot_expected_data_distribution(expected_data, title="Data Difficulty Distribution iGSM-med")