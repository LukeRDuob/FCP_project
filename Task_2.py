import numpy as np
import matplotlib.pyplot as plt
import argparse
import random

def initialize_opinions(population_size):
    #Initialize the population's opinions randomly between 0 and 1

    return np.random.rand(population_size)

def update_opinions(opinions, threshold, beta):

    # Randomly select an individual
    rand_ind = random.randint(0, len(opinions)-1)
    individual_opinion = opinions[rand_ind]
    
    # Randomly select one of its neighbors
    neighbor_rand_ind = random.choice([rand_ind-1, rand_ind+1])
    
    # Ensure boundary conditions
    neighbor_rand_ind = max(0, min(len(opinions)-1, neighbor_rand_ind))
    
    neighbor_opinion = opinions[neighbor_rand_ind]
    
    # Calculate difference of opinions
    diff = abs(individual_opinion - neighbor_opinion)

    # Update opinions if within threshold
    if diff < threshold:
        opinions[rand_ind] = opinions[rand_ind] + beta * (neighbor_opinion - individual_opinion)
        opinions[neighbor_rand_ind] = opinions[neighbor_rand_ind] + beta * (individual_opinion - neighbor_opinion)

    return opinions

def plot_opinions_hist(opinions, timestep, ax):

    bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    ax.hist(opinions, bins=bins)
    ax.set_title(f'Timestep = {timestep}')
    ax.set_xlabel('Opinion')  # Set x-axis label
    ax.set_ylabel('Frequency')  # Set y-axis label


def plot_opinions_scatter(opinions, timestep, ax, beta, threshold):

    x_axis = [timestep] * len(opinions)
    ax.scatter(x_axis, opinions, color = 'red')
    ax.set_title(f'Coupling: {beta}, Threshold: {threshold}')
    ax.set_xlabel('Timestep')  # Set x-axis label
    ax.set_ylabel('Opinion')  # Set y-axis label


def defaunt_main(population_size, threshold, beta, timestep):

    opinions = initialize_opinions(population_size)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ''' Uncomment to see animation'''
    #plt.ion()
    for t in range(timestep):
        
        plot_opinions_hist(opinions, t+1, ax1)
        plot_opinions_scatter(opinions, t+1, ax2, beta, threshold)
        for step in range(timestep):
            update_opinions(opinions, threshold, beta)

        #plt.draw()
        #plt.pause(0.01)
        if t != timestep-1:
            ax1.clear()

    #plt.ioff()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate plots to show how opinions change with interactions')
    
    parser.add_argument('-beta', nargs='?', type=float, default=0.2)
    parser.add_argument('-threshold', nargs='?', type=float, default=0.2)
    parser.add_argument('-population_size', nargs='?', type=int, default=100)
    parser.add_argument('-timestep', nargs='?', type=int, default=100)
    parser.add_argument('-test_defaunt', action = 'store_true')
    parser.add_argument('-defaunt', action = 'store_true')

    args = parser.parse_args()  # Parse the command-line arguments

    beta = args.beta
    threshold = args.threshold
    population_size = args.population_size
    timestep = args.timestep


    if args.defaunt:
        defaunt_main(population_size, threshold, beta, timestep)
    if args.test_defaunt:
        test_defaunt()

if __name__ == '__main__':
    main()

