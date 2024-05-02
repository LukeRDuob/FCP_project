# FCP_project

Requirments:
Install numpy
Install matplotlib.pyplot
Install matplotlib.cm
Install argparse
Install random

Ensure location of file FCP_assignment.py is known.

Access folder that file FCP_assignment.py resides in, then run through terminal. To use a flag, use syntax '-flag_name' and if necessary type down value to be inputted afterwards.

Flags for Task 1:

$ python3 FCP_assignment.py -ising_model #This should run the ising model with default parameters
$ python3 FCP_assignment.py -ising_model -external -0.1 #This should run the ising model with default temperature and an external influence
$ python3 FCP_assignment.py -ising_model -alpha 10 #This should run the ising model with no external influence but with a temperature
$ python3 FCP_assignment.py -test_ising #This should run the test functions associated with the model.

Flags for Task 2:

$ python3 FCP_assignment.py -defuant #This should run the defuant model with default parameters
$ python3 FCP_assignment.py -defuant -beta 0.1 #This should run the defuant model with default threshold and a beta of 0.1.
$ python3 FCP_assignment.py -defuant -threshold 0.3 #This should run the defuant model with a threshold of 0.3
$ python3 FCP_assignment.py -test_defuant #This should run the test functions that you have written.

Flags for Task 3:
$ python3 FCP_assignment.py -network 10 #This should create and plot a random network of size 10
Mean degree: <number>
Average path length: <number>
Clustering co-efficient: <number>
$ python3 FCP_assignment.py -test_network #This should run the test functions that we have provided

Flags for Task 4:
$ python3 FCP_assignment.py -ring_network 10 # This should create a ring network with a range of 1 and a size of 10
$ python3 FCP_assignment.py -small_world 10 #This should create a small-worlds network with default parameters
$ python3 FCP_assignment.py -small_world 10 -re_wire 0.1 #This should create a small worlds network with a re-wiring probability of 0

Flags for Task 5:
$ python3 FCP_assignment.py -defuant -use_network 100 #This should solve the defuant model on a small world network of size 100.

Link to Github repository - https://github.com/LukeRDuob/FCP_project
