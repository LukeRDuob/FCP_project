import numpy as np
import matplotlib.pyplot as plt
import argparse
import random

def initialize_opinions(population_size):
    #Initialize opinions randomly between 0 and 1
    return random.random(population_size)

def update_opinions(opinions, threshold, coupling):

    # Randomly select an individual
    idx = random.randint(0, len(opinions))
    individual_opinion = opinions[idx]
    
    # Randomly select one of its neighbors
    neighbor_idx = random.choice([idx-1, idx+1])
    
    # Ensure boundary conditions
    neighbor_idx = max(0, min(len(opinions)-1, neighbor_idx))
    
    neighbor_opinion = opinions[neighbor_idx]
    
    # Calculate difference of opinions
    diff = abs(individual_opinion - neighbor_opinion)
    
    # Update opinions if within threshold
    if diff <= threshold:
        mean_opinion = (individual_opinion + neighbor_opinion) / 2
        opinions[idx] += coupling * (mean_opinion - individual_opinion)
        opinions[neighbor_idx] += coupling * (mean_opinion - neighbor_opinion)
    
    return opinions

def plot_opinions(opinions, step):
    """Plot opinions."""
    plt.plot(opinions, label=f'Step {step}')
    plt.xlabel('Individuals')
    plt.ylabel('Opinion')
    plt.legend()
    plt.title('Opinion Evolution')
    plt.show()

def simulate_model(population_size, threshold, coupling, num_steps):
    """Simulate the model."""
    opinions = initialize_opinions(population_size)
    plot_opinions(opinions, 0)

    for step in range(1, num_steps + 1):
        opinions = update_opinions(opinions, threshold, coupling)
        plot_opinions(opinions, step)

# Test the model with example parameters
population_size = 100
threshold = 0.2
coupling = 0.1
num_steps = 10

simulate_model(population_size, threshold, coupling, num_steps)
