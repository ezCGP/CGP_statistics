import math
import numpy as np
import itertools
from copy import deepcopy
import pdb

input_count = 2
main_count = 5
output_count = 1
genome_size = input_count + main_count + output_count
base_genome = [None]*genome_size
base_genome[-1*input_count:] = ['input']*input_count


def get_actives(genome):
    actives = [genome[main_count]]
    for node_index in reversed(range(main_count)):
        if node_index in actives:
            # HACK FOR WHILE EXPLORING
            if genome[node_index] is None:
                continue
            for input_node in genome[node_index]:
                if input_node >= 0:
                    actives.append(input_node)

    return list(set(actives))

    
genomes = []
# OUTPUT NODES
for node_index in range(main_count):
    new = deepcopy(base_genome)
    new[main_count] = node_index
    genomes.append(new)


active_count = np.zeros(main_count)
for genome in genomes:
    for active_node in get_actives(genome):
        active_count[active_node] +=1

print(len(genomes), active_count)
print()
# MAIN NODES
for node_index in reversed(range(main_count)):
    choices = np.arange(-1*input_count, node_index)
    input_combos = list(itertools.product(choices, choices)) # returns list of tuples of len 2
    print(len(input_combos))

    new_genomes = []
    for base_genome in genomes:
        for input_nodes in input_combos:
            genome = deepcopy(base_genome)
            genome[node_index] = list(input_nodes) #input nodes is tuple so convert to list
            new_genomes.append(genome)

    genomes = new_genomes
    active_count = np.zeros(main_count) # was main_count
    for genome in genomes:
        for active_node in get_actives(genome):
            active_count[active_node] +=1

    print(node_index, len(genomes), active_count)
    print()

    if node_index == 2:
        special0 = 0
        special1 = 0
        special2 = 0
        special3 = 0
        for genome in genomes:
            # which have 1 active in node 2 but are already active?
            actives = get_actives(genome)
            if (2 in actives):
                special0+=1
                for active in actives:
                    if active == 2:
                        continue
                    if genome[active] is None:
                        continue
                    if 1 in genome[active]:
                        special3 += 1
                        break
                '''
                if (1 in actives):
                    special1+=1
                    if (1 in genome[2]):
                        special2+=1
                        #print(genome)
                        #pdb.set_trace()'''

        print(special0, special1, special2, special3)





def nCk(n,k):
    return int(math.factorial(n)  / (math.factorial(k) * math.factorial(n-k)))
