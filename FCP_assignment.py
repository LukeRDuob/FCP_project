import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import random
import sys


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
		"""
		Initialisation of node object.
		"""	
		self.parent = None
		self.index = number
		self.connections = connections
		self.value = value

	def get_neighbours(self):
		'''
		Returns the neighbouring nodes to the current node as an array.
		'''
		return np.where(np.array(self.connections)==1)[0] #  only contains indexes of nodes that are connected		

class Network: 

	def __init__(self, nodes=None):

		if nodes is None:
			self.nodes = []
		else:
			self.nodes = nodes 

	def get_mean_degree(self):
		"""
		A funtion to find mean degree of all nodes in the network.
		"""
		total_degree = 0
		# sum all degrees together
		for node in self.nodes:
			total_degree += node.connections.count(1)
		# divide sum by number of nodes and return to give mean
		return total_degree / len(self.nodes)
	
	def get_mean_clustering(self):
		"""
		Returns the mean clustering coefficient for the network as a float.
		"""
		sum_coef = 0  # sum of coefficients
		for node in self.nodes:
			sum_coef += self.cluster_coef(node)

		mean_cluster_coef = sum_coef / len(self.nodes)    
		return mean_cluster_coef

	def get_mean_path_length(self):
		"""
		A function that finds the mean path length for all nodes in the network.
		"""
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
		'''
		Returns the mean path length for a given node.
		'''
		total_path_length = 0
		node_count = 0
		# loop through all nodes and sum the path lengths from the original node 
		for node in self.nodes:    
			total_path_length += len(self.breadth_first_search(node, start)) - 1
			node_count += 1
		return total_path_length / (node_count - 1)		

	def get_possible_connections(self, n):
		'''
		Returns the number of possible connections for a given number of neighbours.
		'''
		return n * (n - 1)/2

	def cluster_coef(self, node):
		'''
		Returns the clustering coefficient for a node.
		'''

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
		"""
		Returns the shortest route for a given start and end node.
		"""
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
        # return route
		return [node.value for node in route[::-1]]


	def make_ring_network(self, N, neighbour_range=1):
		'''
		Generates ring network.
		'''
		self.nodes=[]
		#Create empty network first like before
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))
		for (index, node) in enumerate(self.nodes):# Gives every node with their index
			for i in range(1,neighbour_range+1): # loops from 1 to the the neighbour range to find the chosen neighbours
				node.connections[(index+i)%N]=1 
				node.connections[(index-i)%N]=1 # Sets the neighbours either side of it to 1

	def make_small_world_network(self, N, re_wire_prob=0.2):
		"""
		Generates a small world network by re-wiring a standard ring network.
		"""
		neighbour_range=2 
		self.make_ring_network(N, neighbour_range) # Start with ring network with neighbour range 2
		for (index, node) in enumerate(self.nodes): # loops through every node to find every connection
			new_connections=[0 for _ in range(N)] # Create an empty list of zeros for the new set of connections
			emptyconnections=[]
			for (edgeindex, edge) in enumerate(node.connections): 
				if edge==0 and edgeindex!=index: # Finds every empty connection to the the current node that isn't itself
					emptyconnections.append(edgeindex) # appends to empty connections
			for (edgeindex, edge) in enumerate(node.connections): #loops through every connection again	
				if edge==1: # Finds active connections 
					re_wire=random.random() 
					if re_wire<re_wire_prob and len(emptyconnections)>0: #If there are empty connections and the probability is enough
						new_connection_index=emptyconnections[random.randint(0,len(emptyconnections))-1] # selects random empty connection
						self.nodes[edgeindex].connections[index]=0 # Sets the connection on the other node to 0
						new_connections[new_connection_index]=1 # Adds the connection to the new connection list 
						emptyconnections.remove(new_connection_index) # There is now a connection so remover from empty connections 
						emptyconnections.append(edgeindex) # Adds the now empty connection to empty connections
						self.nodes[new_connection_index].connections[index] = 1 # Adds the new connection to the other node
					else:
						new_connections[edgeindex]=1 # Nothing is altered so current connection re-added
				else:
					new_connections[edgeindex]=0
			node.connections=np.array(new_connections)
			
	def plot(self, fig=None, ax=None):
		"""
		Function to plot the network.
		"""
		# create figure if not provided
		if fig==None:
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

def test_networks():
	"""
	Testing ring, one sided and fully connected networks.
	"""
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
	assert(network.get_mean_clustering()==0), network.get_mean_clustering()
	assert(network.get_mean_path_length()==2.777777777777778), network.get_mean_path_length()

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
	assert(network.get_mean_clustering()==0),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==5), network.get_mean_path_length()

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
	assert(network.get_mean_clustering()==1),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==1), network.get_mean_path_length()

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
	still_neighbours = True
	sum = 0
	# while not all neighbours visited
	while still_neighbours:
		# visits neighbours to the left and righ
		for i in range(-1,2):
				x = col + i
				if x >=0 and x <= (len(population)-1) and x != col:
					sum += (population[row,col] * population[row,x])
				if x < 0:
					sum += (population[row,col] * population[row,len(population)+x])
				if x > (len(population)-1):
					sum += (population[row,col] * population[row,x-len(population)])
		# visits neighbours above and below
		for j in range(-1,2):
			y = row + 
			# y is the row of the neighbours
			if y >=0 and y <= (len(population)-1) and y != row:
				sum += (population[row,col] * population[y,col])
			if y > (len(population)-1):
				sum += (population[row,col] * population[y-len(population),col])
			if y < 0:
				sum += (population[row,col] * population[y+len(population),col])
		sum += float(external)*population[row, col]
		still_neighbours = False
	return sum

def ising_step(population, alpha, external=0.0):
	'''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
	'''
	# define variable for row and column
	n_rows, n_cols = population.shape
	row = np.random.randint(0, n_rows)
	col  = np.random.randint(0, n_cols)
	agreement = calculate_agreement(population, row, col, external)
	prob = np.exp(-agreement/float(alpha))

	if agreement > 0 and (random.uniform(0,1) < prob):
		population[row, col] *= -1 # flip occurs if agreement is > 0 under a certain prob
	if agreement < 0:
		population[row, col] *= -1 # flip always occurs if agreement < 0  


def plot_ising(im, population):
	'''
	This function will display a plot of the Ising model.
	'''
	new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
	im.set_data(new_im)
	plt.pause(0.1)

def test_ising():
	'''
	This function will test the calculate_agreement function in the Ising model.
	'''
	print("Testing ising model calculation")
	
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
	assert(calculate_agreement(population,1,1,-10)==14), "Test 9"
	assert(calculate_agreement(population,1,1,10)==-6), "Test 10"
	print("Tests passed")

def createranline(length):
	'''
	This function returns a random line sequence of -1's and 1's of a particular length.
	'''
	temp = []
	for i in range(length):
		ran = random.uniform(0,1)
		if ran > 0.5:
			temp.append(1)
		else:
			temp.append(-1)
	return temp

def createpop(length):
	'''
	This function returns a square grid of a particular size.
	'''
	pop = []
	for j in range(length):
		pop.append(createranline(length))
	return pop


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''
def initialize_opinions(population_size):
    """
    Initialize the population's opinions randomly between 0 and 1.
    """

    return np.random.rand(population_size)

def update_opinions(opinions, threshold, beta):
	'''
	This function should return the updated opinions of the population
	after undergoing some calculations.
	Inputs: opinions (list)
			threshold (float)
			beta (float)
	Returns:
			opinions (list)
	'''
	# Randomly select an individual
	rand_ind = random.randint(0, len(opinions)-1)
	individual_opinion = opinions[rand_ind]
	neighbour_found = False

	# loop to ensure the original invidual isn't selected
	while not neighbour_found:
		# Randomly select one of its neighbors
		neighbour_rand_ind = random.choice([rand_ind-1, rand_ind+1])
		# Ensure boundary conditions
		neighbour_rand_ind = max(0, min(len(opinions)-1, neighbour_rand_ind))
		# break if neighbour is not the original individual
		if neighbour_rand_ind != rand_ind:
			neighbour_found == True
			break	

	neighbour_opinion = opinions[neighbour_rand_ind]
	# Calculate difference of opinions
	diff = abs(individual_opinion - neighbour_opinion)
	# Update opinions if within threshold
	if diff < threshold:
		opinions[rand_ind] = opinions[rand_ind] + beta * (neighbour_opinion - individual_opinion)
		opinions[neighbour_rand_ind] = opinions[neighbour_rand_ind] + beta * (individual_opinion - neighbour_opinion)
		# round values to avoid floating point errors
		opinions[rand_ind] = round(opinions[rand_ind],5)
		opinions[neighbour_rand_ind] = round(opinions[neighbour_rand_ind],5)

	return opinions

def update_opinions_network(opinions, threshold, beta, network):
	'''
	This function should return the updated opinions of the population
	for the networks after undergoing some calculations.
	Inputs: opinions (list)
			threshold (float)
			beta (float)
			network (object)
	Returns:
			opinions (list)
	'''
	# Randomly select an individual
	rand_ind = random.randint(0, len(opinions)-1)
	rand_node = network.nodes[rand_ind]
	individual_opinion = rand_node.value

	# Randomly select one of its neighbors
	number_of_neigbours = len(rand_node.get_neighbours())
	# avoid case of 1 neighbour
	if number_of_neigbours > 0:
		neighbour_rand_ind = random.randint(0, number_of_neigbours-1)
	else:
		neighbour_rand_ind = 0
	# get neighbour
	neighbour = network.nodes[neighbour_rand_ind]
	# get neighbour opinion	
	neighbour_opinion = opinions[neighbour_rand_ind]

	# Calculate difference of opinions
	diff = abs(individual_opinion - neighbour_opinion)
	# Update opinions if within threshold
	if diff < threshold:
		#update opinions list
		opinions[rand_ind] = opinions[rand_ind] + beta * (neighbour_opinion - individual_opinion)
		opinions[neighbour_rand_ind] = opinions[neighbour_rand_ind] + beta * (individual_opinion - neighbour_opinion)	
				# round values to avoid floating point errors
		opinions[rand_ind] = round(opinions[rand_ind], 5)
		opinions[neighbour_rand_ind] = round(opinions[neighbour_rand_ind], 5)
		# update node values
		rand_node.value = opinions[rand_ind]
		neighbour.value = opinions[neighbour_rand_ind]		

	return opinions

def plot_opinions_hist(opinions, timestep, ax):
	'''
	A function that plots a histogram of an opionion against it's frequency.
	'''
    bins= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    ax.hist(opinions, bins=bins)
    ax.set_title(f'Timestep = {timestep}')
    ax.set_xlabel('Opinion')  # Set x-axis label
    ax.set_ylabel('Frequency')  # Set y-axis label


def plot_opinions_scatter(opinions, timestep, ax, beta, threshold):
	'''
	A function that plots a scatter plot of time against the opinions.
	'''
    x_axis = [timestep] * len(opinions)
    ax.scatter(x_axis, opinions, color = 'red')
    ax.set_title(f'Coupling: {beta}, Threshold: {threshold}')
    ax.set_xlabel('Timestep')  # Set x-axis label
    ax.set_ylabel('Opinion')  # Set y-axis label

def plot_mean_op(data, total_time):
	"""
	A function to plot mean opinion against time.
	"""
	mean_fig, mean_ax = plt.subplots(1,1)  # fig for mean opinions
	time = [t for t in range(total_time)]
	mean_ax.plot(time, data)
	mean_ax.set_xlabel("Time(s)")
	mean_ax.set_ylabel("Mean Opinion")

def defuant_main(population_size, network, threshold, beta, timestep):
	'''
	This function governs the main function of the defuant model.
	If a network is called in the flags alonside the defuant model 
	it creates an animation of the network evolving overtime and a 
	graph of the mean opinion against time. If not it creates a 
	scatter plot and histogram after 100 timesteps of the simulation.
	Inputs: population_size (int)
			network (object)
			threshold (float)
			beta (float)
			timestep (int)
	Outputs:
			histogram
			scatter plot
			mean opinion
	'''
	# check if network should be used
	if network == None:
		# defuant on a grid
		opinions = initialize_opinions(population_size)
		fig, (ax1, ax2) = plt.subplots(1, 2)
	else:
		# defuant on a network
		opinions = [node.value for node in network.nodes]	
		fig_animation,ax_animation = plt.subplots(1,1)  # fig for network animation
		ax_animation.set_axis_off()
		means = [] 	#list of mean opinions
		# set beta and threshold values to better visualise the change in opinions
		beta = 0.5
		threshold = 0.2

	plt.ion()
	for t in range(timestep):
		
		if network == None:
			# plot figures
			plot_opinions_hist(opinions, t+1, ax1)
			plot_opinions_scatter(opinions, t+1, ax2, beta, threshold)
		else:
			# plot the network if it exists
			network.plot(fig_animation, ax_animation)
			fig_animation.canvas.draw()
			plt.pause(0.05)

			#get mean opinion
			means.append(get_mean_op(opinions))
		for step in range(timestep):
			# update the opinions by the number of timesteps every timestep
			if network==None:
				# when no network provided
				update_opinions(opinions, threshold, beta)
			else:
				# when network provided
				update_opinions_network(opinions, threshold, beta, network)	

		# clear the figure canvas to ready it for the next frame in the animation
		if t != timestep-1:
			if network == None:
				ax1.clear()
			else:
				ax_animation.clear()
	plt.ioff()

	if network != None:
		# plot mean opinions over time
		plot_mean_op(means, timestep)
	plt.show()

def test_defuant():
	'''
	A function to test that the calculations for the defuant model are correct.
	'''
	print("Testing defuant model")

	assert update_opinions([0.45, 0.55], 0.2, 0.2) == [0.47, 0.53], "defuant 1"
	assert update_opinions([0.05, 0.5, 0.95], 0.5, 0.1) == [0.05, 0.5, 0.95] or [0.095, 0.455, 0.95], "defuant 2"
	assert update_opinions([0.2, 0.25, 0.3], 0.5, 0.5) == [0.2, 0.275, 0.275] or [0.225, 0.225, 0.3] or [0.25, 0.25, 0.25], "defuant 3"
	assert update_opinions([0.2, 0.25, 0.4], 0.2, 0.2) == [0.21, 0.24, 0.4] or [0.2, 0.25, 0.4] or [0.2, 0.28, 0.37], "defuant 4"

	print('Tests Passed')

def get_mean_op(opinions):
	"""
	Plotting mean opinions.
	"""
	mean = sum(opinions) / len(opinions)
	return mean

'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def all_flags():
	"""
	A function to check all flags that could be inputted.
	"""
	
	# create parser to access arguments from the terminal
	parser = argparse.ArgumentParser(description='Opinion dynamics')

	# add arguments for task 1
	parser.add_argument("-ising_model", action = 'store_true')
	
	parser.add_argument("-external", nargs='?', default=0)
	parser.add_argument("-alpha", nargs='?', default=1)
	parser.add_argument("-test_ising", action = 'store_true')

	# add arguments for task 2
	parser.add_argument('-beta', nargs='?', type=float, default=0.2)
	parser.add_argument('-threshold', nargs='?', type=float, default=0.2)
	parser.add_argument('-population_size', nargs='?', type=int, default=100)
	parser.add_argument('-timestep', nargs='?', type=int, default=100)
	parser.add_argument('-test_defuant', action = 'store_true')
	parser.add_argument('-defuant', action = 'store_true')
	
	# add arguments for task 3 
	parser.add_argument("-test_network", action="store_true")
	parser.add_argument("-network", nargs='?', type=int, default=None)
	
	# add arguments for task 4
	parser.add_argument("-ring_network", type=int)
	parser.add_argument("-small_world", type=int)
	parser.add_argument("-re_wire", nargs="?", type=float, default=0.2)

	# add arguements for task 5
	parser.add_argument("-use_network", nargs="?", type=int, default=None)

	# parse args
	args = parser.parse_args()

	# execute flags
	beta = args.beta
	threshold = args.threshold
	population_size = args.population_size
	timestep = args.timestep
	show_ising_model = args.ising_model
	show_test_ising = args.test_ising
	alpha = args.alpha
	external = args.external
	
	
	if show_ising_model:
		population=np.array(createpop(100))
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()
		im = ax.imshow(population, interpolation='none', cmap='RdPu_r')
		for frame in range(100):
			# Iterating single steps 1000 times to form an update
			for step in range(1000):
				ising_step(population, alpha, external)
			print('Step:', frame, end='\r')
			plot_ising(im, population)

	if show_test_ising:
		test_ising()


	if args.network:
		# create network of size 10
		network = Network()
		network.make_random_network(args.network, connection_probability=0.5)
		network.plot()
		plt.show()
		# display relevant information
		print("Mean degree:", network.get_mean_degree())
		print("Average path length:", network.get_mean_path_length())
		print("mean clustering coefficient", network.get_mean_clustering())


	if args.defuant:
		#check if network should be used
		if args.use_network == None:
			# run defuant model with grid
			defuant_main(population_size, None, threshold, beta, timestep)
		else:
			# create small world network
			small_world_network = Network()
			small_world_network.make_small_world_network(args.use_network, 0.2)
			
			# run defuant model
			defuant_main(population_size, small_world_network, threshold, beta, timestep)

	if args.test_defuant:
		test_defuant()
	if args.test_network:
		test_networks()
	if args.ring_network:
		ring__network = Network()
		ring__network.make_ring_network(args.ring_network)
		ring__network.plot()
		plt.show()
	if args.small_world:
		small_world_network = Network()
		small_world_network.make_small_world_network(args.small_world, args.re_wire)
		small_world_network.plot()
		plt.show()


def main():
	all_flags()
	

if __name__=="__main__":
	main()
