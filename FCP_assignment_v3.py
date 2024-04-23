import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import random

class Node:

	def __init__(self, value, number, connections=None):

		self.index = number
		self.connections = connections
		self.value = value

class Network: 

	def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes 

	def get_mean_degree(self):
		#Your code  for task 3 goes here

	def get_mean_clustering(self):
		#Your code for task 3 goes here

	def get_mean_path_length(self):
		#Your code for task 3 goes here

	def make_random_network(self, N, connection_probability=0.5):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	def make_ring_network(self, N, neighbour_range=1):
		#Your code  for task 4 goes here

	def make_small_world_network(self, N, re_wire_prob=0.2):
		#Your code for task 4 goes here

	def plot(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def calculate_agreement(population, row, col, external=0.0):
	'''
	This function should return the extent to which a cell agrees with its neighbours.
	Inputs: population (numpy array)
			row (int)
			col (int)
			external (float)
	Returns:
			change_in_agreement (float)
	'''

	#Your code for task 1 goes here

	return np.random.random() * population

def ising_step(population, external=0.0):
	'''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
	'''
	
	n_rows, n_cols = population.shape
	row = np.random.randint(0, n_rows)
	col  = np.random.randint(0, n_cols)

	agreement = calculate_agreement(population, row, col, external=0.0)

	if agreement < 0:
		population[row, col] *= -1

	#Your code for task 1 goes here

def plot_ising(im, population):
	'''
	This function will display a plot of the Ising model
	'''

    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)

def test_ising():
	'''
	This function will test the calculate_agreement function in the Ising model
	'''

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(population,1,1)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(population,1,1)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(population,1,1)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(population,1,1)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(population,1,1)==4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1,1)==3), "Test 7"
    assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
    assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
    assert(calculate_agreement(population,1,1, -10)==14), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''
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
	
def test_defuant():
	#Your code for task 2 goes here


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

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
	    
if __name__=="__main__":
	main()
