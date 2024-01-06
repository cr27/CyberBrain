# Occipital Lobe (ventral posterior): Visual Processing-
# - Includes primary projection area (V1 or Striate) from LGN of Thalamus & some higher visual areas
# - Divided into separate pathways for Color, Detail, Motion, Depth, etc that move into other lobes
# - Each 0 & 1 correspond to a receptive field for Color, Detail, Motion, and Depth
from collections import deque

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from CMOS_Neuron_Memresistor_Synapse import CMOSNeuron, MemristorSynapse

sr = 1
sc = 1
color = 2


class OccipitalLobe:
    def __init__(self):
        print('')

    def print_info(self, metabolism):

        print('Occipital Lobe Connectivity Graph:')
        print('')

        connections = {
            'Optic Nerve': ['Occipital Lobe'],
            'Hypothalamus': ['Occipital Lobe'],
            'Optic Chiasm': ['Occipital Lobe'],
            'Pons': ['Occipital Lobe'],
            'Medulla Oblongata': ['Occipital Lobe'],
            'Midbrain': ['Occipital Lobe'],
            'Olfactory Bulb': ['Occipital Lobe']
        }

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

        Neurons = 140000000
        Synapses = 27000000000

        print('')
        print('Ratio of Neurons to Synapses in Occipital Lobe:' + '1:192')
        print('Number of Neurons in Occipital Lobe:', Neurons)
        print('Number of Synapses in Occipital Lobe:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 192
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Occipital Lobe:', synapse_sum)
        metabolism.update_synaptic_sums('Occipital Lobe', synapse_sum)
        return G
    # def useColor(self, row, col):
    #     print(color_matrix[row][col])
    #
    # def useDetail(self, row, col):
    #     print(detail_matrix[row][col])
    #
    # def useMotion(self, row, col):
    #     print(motion_matrix[row][col])
    #
    # def useDepth(self, row, col):
    #     print(depth_matrix[row][col])

    def colorPixel(self):

        color_matrix = [1, 1, 1], \
            [1, 1, 0], \
            [1, 0, 1]

        initialColor = color_matrix[sr][sc]

        queue = deque()
        queue.append((sr, sc))
        visited = np.zeros((3, 3), dtype=bool)
        visited[sr, sc] = True

        while queue:
            row, col = queue.popleft()
            # motion_matrix[row][col] = color
            # depth_matrix[row][col] = color
            color_matrix[row][col] = color

            # Check if the pixel above is within bounds and has the same initial color
            if row - 1 >= 0 and color_matrix[row - 1][col] == initialColor and not visited[row - 1][col]:
                queue.append((row - 1, col))
                visited[row - 1][col] = True

            # Check if the pixel below is within bounds and has the same initial color
            if row + 1 < len(color_matrix) and color_matrix[row + 1][col] == initialColor and not visited[row + 1][col]:
                queue.append((row + 1, col))
                visited[row + 1][col] = True

            # Check if the pixel to the left is within bounds and has the same initial color
            if col - 1 >= 0 and color_matrix[row][col - 1] == initialColor and not visited[row][col - 1]:
                queue.append((row, col - 1))
                visited[row][col - 1] = True

            # Check if the pixel to the right is within bounds and has the same initial color
            if col + 1 < len(color_matrix[0]) and color_matrix[row][col + 1] == initialColor and not visited[row][
                col + 1]:
                queue.append((row, col + 1))
                visited[row][col + 1] = True

        print(color_matrix)
        return color_matrix

    def detailPixel(self):
        detail_matrix = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0]
        ]
        initialDetail = detail_matrix[sr][sc]
        queue = deque()
        queue.append((sr, sc))
        visited = np.zeros((3, 3), dtype=bool)
        visited[sr, sc] = True

        while queue:
            row, col = queue.popleft()
            detail_matrix[row][col] = color
            # Check if the pixel above is within bounds and has the same initial color
            if row - 1 >= 0 and detail_matrix[row - 1][col] == initialDetail and not visited[row - 1][col]:
                queue.append((row - 1, col))
                visited[row - 1][col] = True

            # Check if the pixel below is within bounds and has the same initial color
            if row + 1 < len(detail_matrix) and detail_matrix[row + 1][col] == initialDetail and not visited[row + 1][
                col]:
                queue.append((row + 1, col))
                visited[row + 1][col] = True

            # Check if the pixel to the left is within bounds and has the same initial color
            if col - 1 >= 0 and detail_matrix[row][col - 1] == initialDetail and not visited[row][col - 1]:
                queue.append((row, col - 1))
                visited[row][col - 1] = True

            # Check if the pixel to the right is within bounds and has the same initial color
            if col + 1 < len(detail_matrix[0]) and detail_matrix[row][col + 1] == initialDetail and not visited[row][
                col + 1]:
                queue.append((row, col + 1))
                visited[row][col + 1] = True

        print(detail_matrix)
        return detail_matrix

    def depthPixel(self):
        depth_matrix = [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]

        initialDepth = depth_matrix[sr][sc]
        queue = deque()
        queue.append((sr, sc))
        visited = np.zeros((3, 3), dtype=bool)
        visited[sr, sc] = True

        while queue:
            row, col = queue.popleft()
            depth_matrix[row][col] = color
            # Check if the pixel above is within bounds and has the same initial color
            if row - 1 >= 0 and depth_matrix[row - 1][col] == initialDepth and not visited[row - 1][col]:
                queue.append((row - 1, col))
                visited[row - 1][col] = True

            # Check if the pixel below is within bounds and has the same initial color
            if row + 1 < len(depth_matrix) and depth_matrix[row + 1][col] == initialDepth and not visited[row + 1][
                col]:
                queue.append((row + 1, col))
                visited[row + 1][col] = True

            # Check if the pixel to the left is within bounds and has the same initial color
            if col - 1 >= 0 and depth_matrix[row][col - 1] == initialDepth and not visited[row][col - 1]:
                queue.append((row, col - 1))
                visited[row][col - 1] = True

            # Check if the pixel to the right is within bounds and has the same initial color
            if col + 1 < len(depth_matrix[0]) and depth_matrix[row][col + 1] == initialDepth and not visited[row][
                col + 1]:
                queue.append((row, col + 1))
                visited[row][col + 1] = True

        print(depth_matrix)
        return depth_matrix

    def motionPixel(self):
        motion_matrix = [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ]

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

        print(motion_matrix)
        return motion_matrix
