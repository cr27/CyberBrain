import networkx as nx
from matplotlib import pyplot as plt
from Forebrain import Forebrain, BasalGanglia
from OccipitalLobe import OccipitalLobe
from CMOS_Neuron_Memresistor_Synapse import CMOSNeuron,MemristorSynapse
import numpy as np
import matplotlib.pyplot as plt
import random

class Tectum:
    def __init__(self):
        self.excitatory_neuron = None
        self.inhib_neuron = None
        self.time = None #is time considered temporal - yes
        self.amplitude = None
        self.frequency = None


    def Superior_Colliculus(self):
        print(" ")
        print("Collecting Retina Data...")
        import time
        time.sleep(5)
        print("Generating Visual Field...")
        time.sleep(5)
        visualfield = OccipitalLobe()
        e = visualfield.motionPixel()
        f = visualfield.depthPixel()
        ef = visualfield.detailPixel()
        eff = visualfield.colorPixel()
        print(e, f, ef, eff)

    def inferior_colliculus(self):
        print(" ")
        print("Processing Auditory Neurons...")
        print(" ")

        import numpy as np
        import matplotlib.pyplot as plt

        t = np.linspace(0, 1, 1000)
        neg_sin_wave = np.sin(np.pi/2)
        pos_sin_wave = np.sin(3*np.pi/2)
        f = random.choice([neg_sin_wave, pos_sin_wave])

        if f == 1.0:
            print('Excitatory Neurons Activated')
        if f == -1.0:
            print('Inhibitory Neurons Activated')

    def print_info(self, metabolism):
        print('Tectum Connectivity Graph:')
        print('')

        connections = {
            'Superior Colliculus': ['Thalamus', 'Visual Cortex', 'Inferior Colliculus', 'Brainstem'],
            'Inferior Colliculus': ['Thalamus', 'Superior Colliculus', 'Brainstem'],
            'Thalamus': ['Superior Colliculus', 'Inferior Colliculus', 'Cerebral Cortex', 'Brainstem'],
            'Cerebral Cortex': ['Thalamus']
        }

        G = nx.DiGraph()

        G.add_nodes_from(connections.keys())

        for source, targets in connections.items():
            for target in targets:
                G.add_edge(source, target)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)

        plt.show()

        for key in G:
            print(key + ': ' + str(G[key]))

        Neurons = 500000
        Synapses = 20000000

        print(" ")
        print('Ratio of Neurons to Synapses in Tectum:' + '1:40')
        print('Number of Neurons in Tectum:', Neurons)
        print('Number of Synapses in Tectum:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 40
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Tectum:', synapse_sum)
        metabolism.update_synaptic_sums('Tectum', synapse_sum)
        return G

class Tegmentum:
    def __init__(self):
        print('')
    def print_info(self, metabolism):
        self.produce_dopamine = random.randint(0, 1)

        self.dopamine_production = {'dopamine': self.produce_dopamine}

        bg = BasalGanglia()

        self.eye_movement = bg.control_movement_task_setting()

        print('Eye Movement:', self.eye_movement)

        if self.produce_dopamine == 1:
            print('Dopamine in Equilibrium:', self.produce_dopamine)
        else:
            print('Dopamine not in Equilibrium:', self.produce_dopamine)

        motor_connections = {
            'M1': ['Corticospinal tract'],
            'Corticospinal tract': ['Tegmentum', 'Spinal cord'],
            'Tegmentum': ['Red nucleus', 'Vestibular nucleus', 'Reticular formation'],
            'Red nucleus': ['Rubrospinal tract'],
            'Rubrospinal tract': ['Spinal cord'],
            'Vestibular nucleus': ['Vestibulospinal tract'],
            'Vestibulospinal tract': ['Spinal cord'],
            'Reticular formation': ['Reticulospinal tract'],
            'Reticulospinal tract': ['Spinal cord']
        }

        # create an empty directed graph
        F = nx.DiGraph()

        # add nodes for each brain region
        F.add_nodes_from(motor_connections.keys())

        # add edges for each connection in the dictionary
        for source, targets in motor_connections.items():
            for target in targets:
                F.add_edge(source, target)

        # draw the graph using NetworkX
        pos = nx.spring_layout(F)
        nx.draw(F, pos, with_labels=True)

        # show the graph
        plt.show()
        print(" ")
        print('Major Motor Pathways of Tegmentum')
        for key in F:
            print(key + ': ' + str(F[key]))

        print(" ")
        print("Tegmentum Connectivity Graph")
        print(" ")

        connections = {
            'Reticular Formation':['Spinal cord', 'Thalamus', 'Cerebellum'],
            'Red nucleus': ['Cerebellum', 'Spinal cord'],
            'Substantia nigra': ['Basal ganglia', 'Thalamus', 'Brainstem'],
            'Superior colliculus': ['Occipital Lobe', 'Brainstem'],
            'Inferior colliculus': ['Temporal Lobe', 'Brainstem']
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
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)

        # show the graph
        plt.show()

        for key in G:
            print(key + ': ' + str(G[key]))

        Neurons = 1000000
        Synapses = 20000000

        print(" ")
        print('Ratio of Neurons to Synapses in Tegmentum:' + '1:20')
        print('Number of Neurons in Tegmentum:', Neurons)
        print('Number of Synapses in Tegmentum:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 20
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Tegmentum:', synapse_sum)
        metabolism.update_synaptic_sums('Tegmentum', synapse_sum)
        return G
