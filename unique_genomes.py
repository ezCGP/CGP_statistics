'''
I'm having trouble coming up with a formula for the probability that a node in a genome with arity 2 is active.
I think this is because the paths get so complex so quickly. I thought I'd come up with a way to establish
unique genomes and later filter by what is active and what isn't.
'''

### External Modules
import os
import sys
import time
from copy import deepcopy
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pdb

### Local Files


class Genome():
    def __init__(self, main, arity=1, inputs=1, levels_back=None, genome=None):
        self.arity = arity
        self.input_count = inputs
        self.main_count = main
        self.output_count = 1
        self.genome_count = inputs + main + 1
        if levels_back is None:
            self.levels_back = inputs + main
        else:
            self.levels_back = min(levels_back, main+inputs)
        self.genome = [None]*self.genome_count
        self.actives = []

        if genome is None:
            self.initialize_genome()
        else:
            self.genome = genome

        self.get_actives()


    def initialize_genome(self):
        # Input Nodes
        self.genome[-1*self.input_count:] = ["inputs"]*self.input_count

        # Main Nodes
        for node_index in range(self.main_count):
            relative_starting_point = max(-1*self.input_count, node_index-self.levels_back)
            choices = np.arange(relative_starting_point, node_index)
            input_nodes = np.random.choice(choices, size=self.arity, replace=True)
            self.genome[node_index] = list(input_nodes)

        # Output Nodes...always assume 1
        relative_starting_point = max(0, self.main_count-self.levels_back)
        choices = np.arange(relative_starting_point, self.main_count)
        input_node = np.random.choice(choices, size=1, replace=False)
        self.genome[self.main_count] = list(input_node)[0]


    def get_actives(self):
        actives = [self.genome[self.main_count]]
        for node_index in reversed(range(self.main_count)):
            if node_index in actives:
                for input_node in self.genome[node_index]:
                    if input_node >= 0:
                        actives.append(input_node)

        self.actives = list(set(actives))


    def write_as_string(self):
        '''
        to quickly establish uniqueness of a genome, need an easy way to compare them.
        write the node connects into a simple list and express as a string
        '''
        pass

#ting = Genome(main=5, arity=1, inputs=1)


def all_unique_genomes(main, arity=1, inputs=1, levels_back=None):
    '''
    What if I want to make algs by hand?
    '''
    arity = arity
    input_count = inputs
    main_count = main
    output_count = 1
    genome_count = main_count + input_count + output_count

    # assuming the number 'levels_back' is relative to main nodes and then later we'll add input_count to make it easier for indexing
    if levels_back is None:
        levels_back = main_count+input_count
    # going to include input_count manually
    #levels_back += input_count

    base_genome = [None] * genome_count
    genomes = [base_genome]

    # input nodes
    #keep None

    # main nodes
    for node_index in range(main_count):
        relative_starting_point = max(-1*input_count, node_index-levels_back)
        choices = np.arange(relative_starting_point, node_index)
        if arity == 1:
            input_combos = list(itertools.product(choices)) # returns list of tuples of len 1
        if arity == 2:
            input_combos = list(itertools.product(choices, choices)) # returns list of tuples of len 2

        new_genomes = []
        for base_genome in genomes:
            inputs = []
            for input_nodes in input_combos:
                genome = deepcopy(base_genome)
                genome[node_index] = input_nodes # HACKED list(input_nodes) #input nodes is tuple so convert to list
                new_genomes.append(genome)

        genomes = new_genomes
        print(len(genomes))

    # now do the same but for output nodes
    new_genomes = []
    relative_starting_point = max(0, main_count-levels_back)
    choices = np.arange(relative_starting_point, main_count)
    for base_genome in genomes:
        for option in choices:
            genome = deepcopy(base_genome)
            genome[main_count] = option
            new_genomes.append(genome)

    genomes = new_genomes
    print(len(genomes))
    return genomes



def all_unique_counts(main, arity=1, inputs=1, levels_back=None):
    '''
    get full count of active nodes by doing same as all_unique_genomes BUT just get the counts!
    '''
    arity = arity
    input_count = inputs
    main_count = main
    output_count = 1
    genome_count = main_count + input_count + output_count

    # assuming the number 'levels_back' is relative to main nodes and then later we'll add input_count to make it easier for indexing
    if levels_back is None:
        levels_back = main_count + input_count
    # going to include input_count manually
    #levels_back += input_count

    base_genome = [None] * genome_count
    genomes = 1 #[base_genome]
    active_count = {}
    for i in range(main):
        active_count[i] = 0

    # input nodes
    #keep None

    # main nodes
    for node_index in range(main_count):
        relative_starting_point = max(-1*input_count, node_index-levels_back)
        choices = np.arange(relative_starting_point, node_index)
        if arity == 1:
            input_combos = list(itertools.product(choices)) # returns list of tuples of len 1
        if arity == 2:
            input_combos = list(itertools.product(choices, choices)) # returns list of tuples of len 2
        '''
        new_genomes = []
        for base_genome in genomes:
            inputs = []
            for input_nodes in input_combos:
                genome = deepcopy(base_genome)
                genome[node_index] = list(input_nodes) #input nodes is tuple so convert to list
                new_genomes.append(genome)'''

        #genomes = new_genomes
        #print(len(genomes))
        genomes *= len(input_combos)
        print(genomes)

        # for the node_index, all genomes that aren't already

    # now do the same but for output nodes
    relative_starting_point = max(0, main_count-levels_back)
    choices = np.arange(relative_starting_point, main_count)
    '''
    new_genomes = []
    for base_genome in genomes:
        for option in choices:
            genome = deepcopy(base_genome)
            genome[main_count] = option
            new_genomes.append(genome)

    genomes = new_genomes
    print(len(genomes))'''
    genomes *= len(choices)
    print(genomes)

    return genomes


def prob_active_arity1(main, inputs=1, levels_back=None):
    input_count = inputs
    main_count = main
    output_count = 1
    genome_count = main_count + input_count + output_count
    if levels_back is None:
        levels_back = main_count + input_count

    prob_active = np.zeros((main+1,)) #include output node
    prob_active[main] = 1
    #prob_active[main-1] = 1/levels_back
    for i in reversed(range(main)):
        n = min(main, i+levels_back)
        for j in range(i+1, n+1):
            if j == main_count:
                prob_given_j = 1/min(levels_back, main_count) #included 'main' jic levels back was set to None and then later set to main+input_count
            elif (i+1 <= j) and (j <= min(main_count-1, i+levels_back)):
                # honestly could've just done an 'else' statement since there are constraints earlier that force the directly above constraint
                prob_given_j = 1/min(levels_back, input_count+j)
            else:
                # this should never happen because of how for loop is bounded
                prob_given_j = 0
            prob_active[i] += prob_given_j*prob_active[j]

    return prob_active



def prob_active_arity2(main, inputs=1, levels_back=None):
    input_count = inputs
    main_count = main
    output_count = 1
    genome_count = main_count + input_count + output_count
    if levels_back is None:
        levels_back = main_count + input_count

    prob_active = np.zeros((main+1,)) #include output node
    prob_active[main] = 1
    #prob_active[main-1] = 1/levels_back
    for i in reversed(range(main)):
        all_j = np.arange(i+1, min(main, i+levels_back))
        input_combos = list(itertools.product(all_j, all_j))

        for j0, j1 in input_combos:
            for j in [j0,j1]:
                if j == main_count:
                    prob_given_j = 1/min(levels_back, main_count) #included 'main' jic levels back was set to None and then later set to main+input_count
                elif (i+1 <= j) and (j <= min(main_count-1, i+levels_back)):
                    # honestly could've just done an 'else' statement since there are constraints earlier that force the directly above constraint
                    pass








                else:
                    # this should never happen because of how for loop is bounded
                    prob_given_j = 0
                prob_active[i] += prob_given_j*prob_active[j]

    return prob_active


arity=2
inputs=2

for main in range(5,6):
    levels_back = None
    all_genomes_list = all_unique_genomes(main=main, arity=arity, inputs=inputs, levels_back=levels_back)
    all_genomes = []
    for genome in all_genomes_list:
        all_genomes.append(Genome(main=main, arity=arity, inputs=inputs, genome=genome, levels_back=levels_back))

    #del all_genomes_list

    prob_active = np.zeros((main,))
    for main_node in range(main):
        for genome in all_genomes:
            if main_node in genome.actives:
                prob_active[main_node] +=1

    #prob_active /= len(all_genomes)
    print(main, prob_active/len(all_genomes), len(all_genomes))
    #del all_genomes
    print("next")
    all_unique_counts(main=main, arity=arity, inputs=inputs, levels_back=levels_back)
    #print(len(poop))

    #prob_active_test = prob_active_arity1(main=main, inputs=inputs, levels_back=levels_back)
    #print(prob_active_test[:-1])

'''
main = 50
for levels_back in range(1,10):
    prob_active_test = prob_active_arity1(main=main, inputs=inputs, levels_back=levels_back)
    actual = prob_active_test[levels_back-1-inputs]
    est = 2/(levels_back+1)
    print("l=%i: actual %.5f, est %.5f" % (levels_back, actual, est))'''

'''
fig, axes = plt.subplots(1,3, sharey=True, figsize=(15,5))
for i, main in enumerate([10,20,50]):
    standard = prob_active_arity1(main=main, inputs=1, levels_back=None)[:-1]
    experiment0 = prob_active_arity1(main=main, inputs=1, levels_back=main//2)[:-1]
    experiment1 = prob_active_arity1(main=main, inputs=1, levels_back=main//4)[:-1]
    experiment2 = prob_active_arity1(main=main, inputs=1, levels_back=2)[:-1]

    x=np.arange(main)
    axes[i].plot(x, standard, marker="x", label="standard")
    axes[i].plot(x, experiment0, marker="x", label="N/2")
    axes[i].plot(x, experiment1, marker="x", label="N/4")
    axes[i].plot(x, experiment2, marker="x", label="2")
    axes[i].set_title("%i Nodes" % main)

axes[0].set_ylabel("Probability ith node active")
plt.legend()
fig.show()
plt.savefig("probactive_levelsback.png")
'''
exit()


'''
output_4 = []
output_3 = []
output_2 = []
output_1 = []
output_0 = []
for genome in all_genomes_list:
    if genome[main] == 0:
        continue
        output_0.append(genome[:main])

    elif genome[main] == 1:
        continue
        new = deepcopy(genome[:main])
        for i in range(1):
            new[i] = (-99, -99)
        output_1.append(new)

    elif genome[main] == 2:
        new = deepcopy(genome[:main])
        for i in range(2):
            new[i] = (-99, -99)
        output_2.append(new)

    elif genome[main] == 3:
        new = deepcopy(genome[:main])
        for i in range(2):
            new[i] = (-99, -99)
        output_3.append(new)

    elif genome[main] == 4:
        new = deepcopy(genome[:main])
        for i in range(2):
            new[i] = (-99, -99)
        output_4.append(new)


output_4 = np.array([list(x) for x in set(tuple(x) for x in output_4)])
output_3 = np.array([list(x) for x in set(tuple(x) for x in output_3)])
output_2 = np.array([list(x) for x in set(tuple(x) for x in output_2)])
#output_1 = np.array(output_1)
#output_0 = np.array(output_0)
'''

# get all cases where 2 and 3 active ANd 3->1 and 2->
true23 = 0
for genome in all_genomes:
    if (2 in genome.actives) and (3 in genome.actives):
        if (1 in genome.genome[2]) and (1 in genome.genome[3]):
            true23+=1
            print(genome.actives)

true2 = 0
for genome in all_genomes:
    if (2 in genome.actives):
        if (1 in genome.genome[2]):
            true2+=1

true3 = 0
for genome in all_genomes:
    if (3 in genome.actives):
        if (1 in genome.genome[3]):
            true3+=1



'''
could we come up with GP symbolic regression problem to come up with pdf?
include binomial dist or basic discrete uniform with exponential and prod over a range


ORRR What if we can come up with an equation for how many total paths we will have and how 
often a node is active in that path

Verify that this is even the right size
'''
exit()
sim_size = int(1e6)
all_actives = np.zeros((main,))
for i in range(sim_size):
    if i%int(1e5)==0:
        print("%i/%i" % (i,sim_size))
    genome = Genome(main=main, arity=arity, inputs=inputs, levels_back=levels_back)
    for node_index in genome.actives:
        all_actives[node_index] +=1

all_actives /= sim_size
print()
print(prob_active)
print(all_actives) # they match!
