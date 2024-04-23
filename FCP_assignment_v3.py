import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import random

# create class for Queue (used in Breadth-first search)
class Queue:

    def __init__(self):
        self.queue = []

    def push(self, item):
        self.queue.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.queue.pop(0)
        
    def is_empty(self):
          return len(self.queue)==0

class Node:

	def __init__(self, value, number, connections=None):
		self.parent = None
		self.index = number
		self.connections = connections
		self.value = value

	def get_neighbours(self):
		'''returns the neighbouring nodes to the current node as an array'''
		return np.where(np.array(self.connections)==1)[0] #  only contains indexes of nodes that are connected		

class Network: 

	def __init__(self, nodes=None):

		if nodes is None:
			self.nodes = []
		else:
			self.nodes = nodes 

	def get_mean_degree(self):
		"""funtion to find mean degree of all nodes in the network"""
		total_degree = 0
		# sum all degrees together
		for node in self.nodes:
			total_degree += node.connections.count(1)
		# divide sum by number of nodes and return to give mean
		return total_degree / len(self.nodes)
	
	def get_mean_clustering(self):
		"""returns the mean clustering coefficient for the network as a float"""
		sum_coef = 0  # sum of coefficients
		for node in self.nodes:
			sum_coef += self.cluster_coef(node)

		mean_cluster_coef = sum_coef / len(self.nodes)    
		return mean_cluster_coef

	def get_mean_path_length(self):
		"""function that finds the mean path length for all nodes in the network"""
		total = 0
		for node in self.nodes:
			total += self.mean_path_length(node)    
		return total / len(self.nodes)	

	def make_random_network(self, N, connection_probability):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''
		# create attribute for the nodes in the network
		self.nodes = []
		# create N nodes for the network
		for node_number in range(N):
			# assign random value
			value = np.random.random()
			# set all connections to 0 
			connections = [0 for _ in range(N)]
			# add node to list
			self.nodes.append(Node(value, node_number, connections))
		
		# generate random connections based off probability
		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					# set current index position in the connections list to 1
					node.connections[neighbour_index] = 1
					# set index of other neighbour to 1 to ensure the network is undirected
					self.nodes[neighbour_index].connections[index] = 1	

	def mean_path_length(self, start):
		'''returns the mean path length for a given node'''
		total_path_length = 0
		node_count = 0
		# loop through all nodes and sum the path lengths from the original node 
		for node in self.nodes:    
			total_path_length += len(self.breadth_first_search(node, start)) - 1
			node_count += 1
		return total_path_length / (node_count - 1)		

	def get_possible_connections(self, n):
		'''returns the number of possible connections for a given number of neighbours'''
		return n * (n - 1)/2

	def cluster_coef(self, node):
		'''returns the clustering coefficient for a node'''

		#find neighbours and 
		neighbours = node.get_neighbours()
		possible_connects = self.get_possible_connections(len(neighbours))
		used = []  # list of used nodes to avoid counting the same connection twice 
		connects = 0  # counts number of connections
        # iterate through possible connections checking if neighbours link to each other
		for neighbour in neighbours:
            # get neighbours of the neigbour
			other_neighbours = self.nodes[neighbour].get_neighbours()
            # loop through the neighbour's neighbours
			for other_neighbour in other_neighbours:
                # check if this node is also a neighbour of the original node
				if other_neighbour in neighbours:
					if [neighbour, other_neighbour] not in used:
						connects += 1
						used.append([other_neighbour, neighbour])
		# avoid dividing by 0 in cases when there are no possible connections 
		if possible_connects != 0: 
			coef = connects / possible_connects
		else: 
			coef = 0	   
		return coef
	
	def breadth_first_search(self, goal, start_node):
		"""returns the shortest route for a given start and end node"""
        # create search queue to keep track of the route
		search_queue = Queue()
		search_queue.push(start_node)
        # list of visited nodes to ensure nodes are only visited once
		visited = []

		while not search_queue.is_empty():
            # check the node at the top of the queue
			node_to_check = search_queue.pop()
			if node_to_check == goal:
				break
            # loop through the nodes connected to the current node
			for neighbour_index in node_to_check.get_neighbours():
				neighbour = self.nodes[neighbour_index]
				if neighbour_index not in visited:
					search_queue.push(neighbour)
					visited.append(neighbour_index)
					neighbour.parent = node_to_check
        # retrace steps to obtian route
		node_to_check = goal
		start_node.parent = None
		route = []
		# retrace until no more parent nodes
		while node_to_check.parent:
			route.append(node_to_check)
			node_to_check = node_to_check.parent
		route.append(node_to_check)
        # output route
		return [node.value for node in route[::-1]]


	def make_ring_network(self, N, neighbour_range=1):
		#Your code  for task 4 goes here
		print("")

	def make_small_world_network(self, N, re_wire_prob=0.2):
		#Your code for task 4 goes here
  		print("")
		  
	def plot(self):
		"""function to plot the network"""
		# create figure
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()  # remove axes

		# create ring to help position the nodes 
		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])
		
		# find coodinates for the nodes 
		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			# draw circle to represent each node
			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			# find neighbouring nodes for the current node to connect to 
			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					# find location of neibouring node
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)
					# draw edge 
					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')
		plt.show()

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
	assert(network.get_clustering()==0), network.get_mean_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_mean_path_length()

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
	assert(network.get_clustering()==0),  network.get_mean_clustering()
	assert(network.get_path_length()==5), network.get_mean_path_length()

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

def defuant_main():
	#Your code for task 2 goes here
	print("")

def test_defuant():
	#Your code for task 2 goes here
	print("")


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def task_3_parse():
	"""function to check for flags and carry out correct processes """
	# create parser to access arguments from the terminal
	parser = argparse.ArgumentParser(description='Opinion dynamics')
	# add arguments for task 3 
	parser.add_argument("-test_network", action="store_true")
	parser.add_argument("-network", nargs='?', type=int, default=6)

	args = parser.parse_args()

	# check if the flags are used
	if args.test_network:
		print("testing task 3")
		# run test function

	# set network size
	network_size = args.network  # default size
	
	return network_size

def main():
		
	node_number = task_3_parse()
	network = Network()
	network.make_random_network(node_number, 0.6)
	network.plot()
	
	
	print("Mean degree:", network.get_mean_degree())
	print("Average path length:", network.get_mean_path_length())
	print("mean clustering coefficient", network.get_mean_clustering())

if __name__=="__main__":
	main()