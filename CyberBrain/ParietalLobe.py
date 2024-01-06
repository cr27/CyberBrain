import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from CMOS_Neuron_Memresistor_Synapse import CMOSNeuron, MemristorSynapse
from FrontalLobe import FrontalLobe
import random
touch, vibration, pressure, pain, temperature, position = random.randint(0, 1), \
    random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)

class ParietalLobe():
    def __init__(self):
        print("")
    def print_info(self, metabolism):
        print("Parietal Lobe Connectivity Graph:")
        print("")
        connections = {
            'Parietal Lobe': ['S1', 'Frontal Lobe'],
            'S1': ['M1', 'Parietal Lobe'],
            'M1': ['S1', 'Frontal Lobe',],
            'Frontal Lobe': ['M1','Parietal Lobe']
        }

        # create an empty directed graph
        G = nx.DiGraph()

        # add nodes for each brain region
        G.add_nodes_from(connections.keys())

        # add edges for each connection in the dictionary
        for source, targets in connections.items():
            for target in targets:
                G.add_edge(source, target)

        # draw the graph using NetworkX
        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True)

        # Show the connectivity graph
        plt.show()

        for key in G:
            print(key + ': ' + str(G[key]))

        Neurons = 120000000
        Synapses = 6000000000

        print("")
        print('Ratio of Neurons to Synapses in Parietal Lobe: 1:50')
        print('Number of Neurons in Parietal Lobe:', Neurons)
        print('Number of Synapses in Parietal Lobe:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 50
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Parietal Lobe:', synapse_sum)
        metabolism.update_synaptic_sums('Parietal Lobe', synapse_sum)
        return G

    def receptors(self):
        if touch == 1:
            print("Touch Receptor Activated")
        else:
            print("False")

        if vibration == 1:
            print("Vibration Receptor Activated")
        else:
            print("False")

        if pressure == 1:
            print("Pressue Receptor Activated")
        else:
            print("False")

        if pain == 1:
            print("Pain Receptor Activated")
        else:
            print("False")

        if temperature == 1:
            print("Temperature Receptor Activated")
        else:
            print("False")

        if position == 1:
            print("Position Receptor Activated")
        else:
            print("False")


    def S1(self):
        print("")
        print(self.receptors())
        print("")
        print("S1 Connectivity Graph:")
        print("")
        connections = {
            'Thalamus': ['S1'],
            'S1': ['Thalamus', 'M1']
        }

        # create an empty directed graph
        G = nx.DiGraph()

        # add nodes for each brain region
        G.add_nodes_from(connections.keys())

        # add edges for each connection in the dictionary
        for source, targets in connections.items():
            for target in targets:
                G.add_edge(source, target)

        # draw the graph using NetworkX
        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True)

        # Show the connectivity graph
        plt.show()

        for key in G:
            print(key + ': ' + str(G[key]))

        Neurons = 3000000
        Synapses = 10 ** 12

        print(" ")
        print('Ratio of Neurons to Synapses in S1:' + '333000')
        print('Number of Neurons in S1:', Neurons)
        print('Number of Synapses in S1:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')
        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 333000
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in S1:', synapse_sum)
        return G
    # S1- primary somatosensory cortex + primary motor cortex - M1
    def penfieldMap(self):
        print(self.S1)
        my_primarymotorCortex = FrontalLobe()
        my_primarymotorCortex.M1()


    def mirrorNeuronSystem(self):
        import heapq
        count = 0
        seen_set = set()  # initialize an empty set
        duplicates_heap = []  # initialize an empty heap

        while True:
            user_input = input("Enter a value (or 'exit' to quit): ")

            if user_input.lower() == "exit":
                break  # exit the loop if user enters 'exit'

            if user_input in seen_set:
                heapq.heappush(duplicates_heap, user_input)  # add the duplicate to the heap
            else:
                seen_set.add(user_input)  # add the user's input to the set

        print("Seen set:", seen_set)
        print("Mirror Neurons Active:", duplicates_heap)

    # Parietal Lobe also includes higher visual areas of “Where/How” pathway
    def higherVisualAreaWhereHow(self):
        print("")
    #- e.g. Canonical Cells, that respond to “affordances” of object (how it can be handled, used)
    def canonicalCells(self):
        print("")