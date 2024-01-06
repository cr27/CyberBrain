import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from CMOS_Neuron_Memresistor_Synapse import CMOSNeuron,MemristorSynapse
import random
touch, vibration, pressure, pain, temperature, position = random.randint(0, 1), \
    random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)
from collections import deque


class medulla_oblongata:
    def __init__(self):
        self.bloodpressure = None
        self.heartrate = None
        self.breathing = None
        self.digestion = None
        self.coughing = None
        self.sneezing = None
        self.vomiting = None

    def set_vitals(self, bloodpressure, heartrate, breathing, digestion):
        self.bloodpressure = bloodpressure
        self.heartrate = heartrate
        self.breathing = breathing
        self.digestion = digestion

    def set_reflexes(self, coughing, sneezing, vomiting):
        self.coughing = coughing
        self.sneezing = sneezing
        self.vomiting = vomiting

    def print_info(self, metabolism):
        print("")
        print("Blood Pressure:", self.bloodpressure)
        print("Heart Rate:", self.heartrate)
        print("Breathing:", self.breathing)
        print("Digestion:", self.digestion)
        print("Coughing:", self.coughing)
        print("Sneezing:", self.sneezing)
        print("Vomiting:", self.vomiting)
        print('')

        print('Medulla Oblongata Connectivity Graph:')
        print('')

        connections = {
            'Spinal Cord': ['Medulla Oblongata'],
            'Pons': ['Medulla Oblongata'],
            'Cerebellum': ['Medulla Oblongata'],
            'Hypothalamus': ['Medulla Oblongata'],
            'Thalamus': ['Medulla Oblongata']}

        G = nx.DiGraph()

        G.add_nodes_from(connections.keys())

        for source, targets in connections.items():
            for target in targets:
                G.add_edge(source, target)

        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True)

        plt.show()

        for key in G:
            print(key + ': ' + str(G[key]))

        Neurons = 10000000
        Synapses = 200000000

        print('')
        print('Ratio of Neurons to Synapses in Medulla Oblongata: 1:' + '1:20')
        print('Number of Neurons in Medulla Oblongata:', Neurons)
        print('Number of Synapses in Medulla Oblongata:', Synapses)
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
        print('Sum of synaptic weights in Medulla Oblongata:', synapse_sum)
        metabolism.update_synaptic_sums('Medulla Oblongata', synapse_sum)
        return G
class Pons:
    def __init__(self):
        self.sleep = None
        self.respiration = None
        self.facialmovement = None

    def set_sleep(self, value):
        self.sleep = value

    def set_respiration(self, value):
        self.respiration = value

    def set_facialmovement(self, value):
        self.facialmovement = value

    def print_info(self, metabolism):
        print('')
        print("Sleep:", self.sleep)
        print("Respiration:", self.respiration)
        print("Facial Movement:", self.facialmovement)
        print('')

        print('Pons Connectivity Graph:')
        print('')

        connections = {
            'Cerebellum': ['Pons'],
            'Medulla Oblongata': ['Pons'],
            'Midbrain': ['Pons'],
            'Thalamus': ['Pons'],
            'Cerebral Cortex': ['Pons']}

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

        # show the graph
        plt.show()

        for key in G:
            print(key + ': ' + str(G[key]))

        Neurons = 95000000
        Synapses = 1.5 * 10 ** 9

        print('')
        print('Ratio of Neurons to Synapses in Pons:' + '1:15')
        print('Number of Neurons in Pons:', Neurons)
        print('Number of Synapses in Pons:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 15
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Pons:', synapse_sum)
        metabolism.update_synaptic_sums('Pons', synapse_sum)
        return G

class cerebellum:
    def __init__(self):
        self.coordinatemovement = None
        self.motorcoordination = None
        self.balance = None
        self.posture = None

    def motion_matrix_stimuli(self):
        import random
        from collections import deque
        import numpy as np

        motion_matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]

        positions = random.sample(range(9), 3)

        for position in positions:
            row = random.randint(0, 2)
            col = random.randint(0, 2)
            motion_matrix[row][col] = 1

        for row in motion_matrix:
            print(row)

        sr = 1
        sc = 1
        color = 2

        initialMotion = motion_matrix[sr][sc]
        queue = deque()
        queue.append((sr, sc))
        visited = np.zeros((3, 3), dtype=bool)
        visited[sr, sc] = True

        while queue:
            row, col = queue.popleft()
            motion_matrix[row][col] = color
            # Check if the pixel above is within bounds and has the same initial color
            if row - 1 >= 0 and motion_matrix[row - 1][col] == initialMotion and not visited[row - 1][col]:
                queue.append((row - 1, col))
                visited[row - 1][col] = True

            # Check if the pixel below is within bounds and has the same initial color
            if row + 1 < len(motion_matrix) and motion_matrix[row + 1][col] == initialMotion and not visited[row + 1][
                col]:
                queue.append((row + 1, col))
                visited[row + 1][col] = True

            # Check if the pixel to the left is within bounds and has the same initial color
            if col - 1 >= 0 and motion_matrix[row][col - 1] == initialMotion and not visited[row][col - 1]:
                queue.append((row, col - 1))
                visited[row][col - 1] = True

            # Check if the pixel to the right is within bounds and has the same initial color
            if col + 1 < len(motion_matrix[0]) and motion_matrix[row][col + 1] == initialMotion and not visited[row][
                col + 1]:
                queue.append((row, col + 1))
                visited[row][col + 1] = True
        return motion_matrix

    def reflexes(self):
        # The cerebellum plays a role in classical conditioning of discrete responses such as eye-blink and limb
        # flexion.

        matrix_result = self.motion_matrix_stimuli()
        print(" ")
        for row in matrix_result:
            print(row)
        motion_stimuli_intensity = 0
        for i in range(len(matrix_result)):
            for j in range(len(matrix_result[0])):
                motion_stimuli_intensity += matrix_result[i][j]
        print(" ")
        if motion_stimuli_intensity > 10:
            print("Motion_stimuli_intensity illicits an eye blink:", motion_stimuli_intensity)
        if motion_stimuli_intensity <= 10:
            print("Motion_stimuli_intensity illicits no response:", motion_stimuli_intensity)
        return matrix_result, motion_stimuli_intensity

    def set_coordinatemovement(self, value):
        self.coordinatemovement = value

    def set_motorcoordination(self, value):
        self.motorcoordination = value

    def set_balance(self, value):
        self.balance = value

    def set_posture(self,value):
        self.posture = value

    def print_info(self, metabolism):
        print('')
        print("Coordinate Movement:", self.coordinatemovement)
        print("Motor Coordination:", self.motorcoordination)
        print("Balance:", self.balance)
        print("Posture:", self.posture)
        print('')

        print('Cerebellum Connectivity Graph: ')
        print('')

        connections = {
            'Cerebral Cortex': ['Cerebellum'],
            'Brainstem': ['Cerebellum'],
            'Spinal Cord': ['Cerebellum'],
            'Thalamus': ['Cerebellum'],
            'Superior Colliculus': ['Cerebellum'],
            'Red Nucleus': ['Cerebellum']
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

        # show the graph
        plt.show()

        for key in G:
            print(key + ': ' + str(G[key]))

        Neurons = 6.9 * 10 ** 10
        Synapses = 1.4 * 10 ** 12

        print('')
        print('Ratio of Neurons to Synapses in Cerebellum:' + '1:20')
        print('Number of Neurons in cerebellum:', Neurons)
        print('Number of Synapses in Cerebellum:', Synapses)
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
        print('Sum of synaptic weights in Cerebellum:', synapse_sum)
        metabolism.update_synaptic_sums('Cerebellum', synapse_sum)
        return G