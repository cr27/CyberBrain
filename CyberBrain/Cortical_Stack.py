import numpy as np
import networkx as nx
from CMOS_Neuron_Memresistor_Synapse import CMOSNeuron,MemristorSynapse


class Cortical_Stack:
    def __init__(self):
        print("")

    def Peripheral_Nervous_System(self):
        Neurons = 10000
        Synapses = 1000000000
        print('Ratio of Neurons to Synapses in Peripheral Nervous System:' + '1:100000')
        print('Number of Neurons in Peripheral Nervous System:', Neurons)
        print('Number of Synapses in Peripheral Nervous System:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 100000
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Peripheral Nervous System:', synapse_sum)

class Neurome():

    def Immune_System(self):
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 400
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Immune System:', synapse_sum)

    def Excretory_System(self):
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 400
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Excretory System:', synapse_sum)

    def Circulatory_System(self):
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 400
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Circulatory System:', synapse_sum)

    def Respiratory_System(self):
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 400
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Respiratory System:', synapse_sum)

    def Digestive_System(self):
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 4000
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Digestive System:', synapse_sum)

    def Integumentary_System(self):
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 400
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Integumentary System:', synapse_sum)

    def Endocrine_System(self):
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 400
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Endocrine System:', synapse_sum)

    def Musculoskeletal_System(self):
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 1000
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in musculoskeletal system:', synapse_sum)

    def Neurome_print_connections(self):

        import networkx as nx
        import matplotlib.pyplot as plt

        # Create a directed graph
        G = nx.DiGraph()

        # Define the connections between systems
        connections = {
            'Neurome.Immune_System()': ['Neurome.Excretory_System()'],
            'Neurome.Excretory_System()': ['Neurome.Circulatory_System()'],
            'Neurome.Circulatory_System()': ['Neurome.Respiratory_System()'],
            'Neurome.Respiratory_System()': ['Neurome.Digestive_System()'],
            'Neurome.Digestive_System()': ['Neurome.Integumentary_System()'],
            'Neurome.Integumentary_System()': ['Neurome.Endocrine_System()'],
            'Neurome.Endocrine_System()': ['Neurome.Musculoskeletal_System()']
        }

        # Add nodes and edges to the graph
        for system, connected_systems in connections.items():
            G.add_node(system)
            G.add_edges_from([(system, connected_system) for connected_system in connected_systems])

        # Draw the graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=10, font_weight='bold',
                edge_color='gray', arrowsize=10)
        plt.title('Connectivity Graph of Neurome Systems')
        plt.show()









