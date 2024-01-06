# Motor Cortex, Language Production, and Strategy#
##################################################
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from TemporalLobe import TemporalLobe
from binarytree import Node, print_tree_preorder
from BST import BST
from Forebrain import LimbicSystem
from LinkedList import LinkedList, LinkedListNode
from CMOS_Neuron_Memresistor_Synapse import CMOSNeuron,MemristorSynapse

import random

# generate a random 0 or 1
random_number = round(random.random())

# print the result
print(random_number)


class FrontalLobe():
    def __init__(self):
        print("")

    def print_info(self, metabolism):
        self.my_list = LinkedList()

        print('FrontalLobe Connectivity Graph:')
        print(" ")

        connections = {
            'Frontal Lobe': ['Cingulate cortex','Parietal Lobe'],
            'Frontal Lobe': ['Cingulate cortex', 'Amygdala', 'Hippocampus'],
            'Cingulate cortex': ['Frontal Lobe', 'Amygdala','Hippocampus'],
            'Parietal Lobe': ['Frontal Lobe', 'M1'],
            'M1': ['Parietal Lobe', 'Basal ganglia'],
            'Basal ganglia': ['M1', 'Thalamus'],
            'Thalamus': ['Basal ganglia', 'M1']
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

        Neurons = 170000000
        Synapses = 100000000000

        print('')
        print('Ratio of Neurons to Synapses in Frontal Lobe:' + '1:588')
        print('Number of Neurons in Frontal Lobe:', Neurons)
        print('Number of Synapses in Frontal Lobe:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 588
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Frontal Lobe:', synapse_sum)
        metabolism.update_synaptic_sums('Frontal Lobe', synapse_sum)
        return G

    def M1(self):
        import numpy as np
        import matplotlib.pyplot as plt

        # Define the parameters of the signal
        num_samples = 1000
        duration = 1.0
        sample_rate = num_samples / duration

        # Generate random amplitudes, frequencies, and phases for the sine waves
        amplitudes = np.random.uniform(low=0.1, high=1.0, size=10)
        frequencies = np.random.uniform(low=1.0, high=10.0, size=10)
        phases = np.random.uniform(low=0.0, high=2 * np.pi, size=10)

        # Generate the time axis
        time = np.linspace(0, duration, num_samples, endpoint=False)

        # Generate the signal by adding up multiple sine waves
        signal = np.zeros(num_samples)
        for i in range(len(amplitudes)):
            signal += amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * time + phases[i])

        # Plot the signal
        # plt.plot(time, signal)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.show()

        print('')
        print('Primary Motor Cortex Connectivity Graph')
        print('')

        connections = {
            'M1': ['S1', 'Frontal Lobe', 'Spinal Cord'],
            'S1': ['M1'],
            'Spinal Cord': ['M1']
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

        Neurons = 1000000000
        Synapses = 20000000000000

        print('')
        print('Ratio of Neurons to Synapses in M1:' + '1:20000')
        print('Number of Neurons in M1:', Neurons)
        print('Number of Synapses in M1:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 20000
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in M1:', synapse_sum)
        return G

    def brocasArea(self):
        signal = TemporalLobe()
        tokens, tokens1, tokens2 = signal.wernickesArea()
        print(tokens)
        print(tokens1)
        print(tokens2)

    def delayedGratification(self):
        print(
            'Delayed gratification refers to the ability to resist the temptation of an immediate reward in order to obtain a larger or more valuable reward in the future. '
            'This is often measured in the context of the "marshmallow test," '
            'in which children are offered a small reward immediately (e.g. one marshmallow), '
            'or a larger reward if they can wait for a period of time (e.g. two marshmallows).',
            end='\n')

        user_input = ""

        while user_input != "one marshmallow" and user_input != "two marshmallows":
            user_input = input(
                "Would you like to receive one marshmallow immediately or two if you wait a longer period of time?")
            if user_input != "one marshmallow" and user_input != "two marshmallows":
                print("Please enter either 'one marshmallow' or 'two marshmallows'.")

        if user_input == "one marshmallow":
            print("You have chosen to receive one marshmallow immediately.")
        else:
            print("You have chosen to receive two marshmallows after waiting." + "Delayed Gratification Enabled")

    ################################################################
    # making a decision on which direction to go based on a threat.#
    ################################################################
    ## must modify code to also implement a punishment/reward for directions chosen -
    ## Pos. for taking path that leads further from threat, Neg. for taking path that leads closer to threat
    ## Animal generates his movement via strings, your movement is generated via input to the console, one set
    ##contains your movement strings, another set contains animal generated movement strings
    def planning(self):
        import random
        animal_movement = {'Up', 'Down', 'Left', 'Right'}
        speed = 0
        print("")
        print("Lion is Chasing you...")
        text2 = "Enter directions to go to avoid an imminent threat, 'Up', 'Down,' 'Left,' or 'Right'"
        path = LinkedList()
        consciousness_input = input(text2)
        path.append(consciousness_input)
        path.print_list()
        while consciousness_input != "quit":
            random_string = random.choice(list(animal_movement))
            consciousness_input = input(text2)
            path.append(consciousness_input)
            if consciousness_input == random_string:
                speed+=1
                print("The lion has chosen the same path, it gets closer to you, you run faster to get further away "
                      "from the lion,", 'your movement increaseses to:', speed,'you are being negativily reinforced')
            if consciousness_input != random_string:
                print("The lion has not chosen the same path, it gets further away from you", "you are being",
"positively reinforced")
            print("If you wish to stop making directions, type 'quit' ")
            if consciousness_input == "quit":
                print('You have chosen to stop making movement')
                print(" ")
                print(" Your movement patterns were: ")
                path.print_list()
                break;


        # random.randint(0, 1)


    def selfControl(self):
        print(
            'Self-control, on the other hand, refers to a broader set of abilities that involve regulating one\'s behavior, thoughts, and emotions in order to achieve a goal or conform to a standard or norm. '
            'Self-control can involve resisting not only immediate rewards, but also other types of temptation, distraction, or impulses that might interfere with one\'s goals or values.',
            end='\n')

        signal = LimbicSystem()
        root = Node(signal.hippocampus())
        # bughuntingcode
        print(root.value)

        # choose a number test that builds upon marshmallow test logic
        # minutes willing to wait test - tests how many minutes one wants to wait for an even bigger reward
        print('choice tree for self-control activating...')
        print('Enter an integer equivalent to the amount of time you want to wait for an even bigger reward based on '
              'the conditions you will be given')
        print("Enter a value greater than", root.value)
        consciousness_input = input(
            "Conscious long-term/short-term memory retrieval in use... if you wish to exercise self control still "
            "enter a value greater than current memory weight")
        root.left_child = Node(int(consciousness_input))
        if root.left_child.value > root.value:
            print('proceed..., maintaining self-control...level of self-control increasing')
        ###################################################################################
        consciousness_input = input(
            "Conscious long-term/short-term memory retrieval in use... if you wish to exercise self control still "
            "enter a value greater than current memory weight")
        print("Enter a value greater than", root.left_child.value)
        root.right_child = Node(int(consciousness_input))
        if root.right_child.value > root.value + root.left_child.value:
            print('proceed..., maintaining self-control...level of self-control increasing')
        ####################################################################################
        consciousness_input = input(
            "Conscious long-term/short-term memory retrieval in use... if you wish to exercise self control still "
            "enter a value greater than current memory weight")
        print("Enter a value greater than", root.right_child.value)
        root.left_child.left_child = Node(int(consciousness_input))
        if root.left_child.left_child.value > root.right_child.value:
            print('proceed..., maintaining self-control...level of self-control increasing')
        ###################################################################################
        consciousness_input = input(
            "Conscious long-term/short-term memory retrieval in use... choosing level of self control")
        print("Enter a value greater than", root.left_child.left_child.value)
        root.left_child.right_child = Node(int(consciousness_input))
        if root.left_child.right_child.value > root.left_child.left_child.value:
            print('proceed..., maintaining self-control...level of self-control increasing')
        ####################################################################################
        consciousness_input = input(
            "Conscious long-term/short-term memory retrieval in use... choosing level of self control")
        print("Enter a value greater than", root.left_child.right_child.value)
        root.right_child.left_child = Node(int(consciousness_input))
        if root.right_child.left_child.value > root.left_child.right_child.value:
            print('proceed..., maintaining self-control...level of self-control increasing')
        ####################################################################################
        consciousness_input = input(
            "Conscious long-term/short-term memory retrieval in use... choosing level of self control")
        print("Enter a value greater than", root.right_child.left_child.value)
        root.right_child.right_child = Node(int(consciousness_input))
        if root.right_child.right_child.value > root.right_child.left_child.value:
            print("Self Control Maxed out")

        print_tree_preorder(root)
        # output is based on user input - choices

    # automated and learned are the rules, but rules can also be removed/changed
    # def culturalRules(self):
    #     cultural_rules = {"Respect elders", "Don't interrupt others while they are speaking",
    #                       "Take off your shoes before entering a home"}
    #     print("Do you need to change cultural rules" + "Type Yes and Enter the rule or Type "
    #     "Anything Else for No")
    #     change_cultural_rules = input()
    #     if change_cultural_rules == "Yes":
    #         print("Enter new rule")
    #     change_cultural_rules = input()
    #     cultural_rules.add(change_cultural_rules)
    #     print(cultural_rules)
    def cultural_rules(self):
        rules = {"Respect elders", "Don't interrupt others while they are speaking",
                 "Take off your shoes before entering a home"}
        print("Current cultural rules: ")
        for rule in rules:
            print("- " + rule)
        print("Do you want to add a new cultural rule? Type 'Yes' or anything else to exit.")
        user_input = input().lower()
        if user_input == "yes":
            new_rule = input("Enter the new rule: ")
            rules.add(new_rule)
            print("New rule added successfully!")
        else:
            print("No new rule added.")
        print("Updated cultural rules: ")
        for rule in rules:
            print("- " + rule)

    def emotionalExpression(self):
        emotions = {
            "happy": 0.8,
            "sad": 0.6,
            "angry": 0.9,
            "fearful": 0.7,
            "surprised": 0.5
        }

        print(emotions)
