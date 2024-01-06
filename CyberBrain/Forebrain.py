import networkx as nx
import matplotlib.pyplot as plt
from Hindbrain import medulla_oblongata, Pons, cerebellum
from CMOS_Neuron_Memresistor_Synapse import CMOSNeuron, MemristorSynapse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
from tqdm.notebook import tqdm

warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
# from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


class Forebrain:
    def __init__(self):
        print('Forebrain Connectivity Graph:')
        print('')
        G = nx.DiGraph()

        G.add_nodes_from(['olfactory cortex', 'hippocampus', 'thalamus', 'basal ganglia', 'amygdala', 'hypothalamus',
                          'basal forebrain'])

        G.add_edge('olfactory cortex', 'hippocampus')
        G.add_edge('olfactory cortex', 'amygdala')
        G.add_edge('hippocampus', 'thalamus')
        G.add_edge('thalamus', 'basal ganglia')
        G.add_edge('thalamus', 'cortex')
        G.add_edge('basal ganglia', 'cortex')
        G.add_edge('amygdala', 'hypothalamus')
        G.add_edge('hypothalamus', 'thalamus')
        G.add_edge('basal forebrain', 'thalamus')
        G.add_edge('basal forebrain', 'cortex')

        for key in G:
            print(key + ': ' + str(G[key]))

        nx.draw(G, with_labels=True)
        plt.show()
        Neurons = 16000000000
        Synapses = 600000000000
        print('')
        print('Ratio of Neurons to Synapses in Forebrain:' + '1:37')
        print('Number of Neurons in Forebrain:', Neurons)
        print('Number of Synapses in Forebrain:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 37
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Forebrain:', synapse_sum)

class Thalamus:
    def __init__(self):
        self.sleep = None
        self.wakefulness_homeostasis = 0.5

    def print_info(self, metabolism):
        print("")
        print("Wakefuleness_homeostasis:", self.wakefulness_homeostasis)
        print("Sleep:", self.sleep)

        print("")
        print('Thalamus Connectivity Graph:')
        print('')

        self.connections = {
            'M1': ['S1', 'A1'],
            'S1': ['M1', 'A1', 'V1'],
            'A1': ['M1', 'S1', 'V1'],
            'V1': ['S1', 'A1']
        }

        G = nx.DiGraph()

        G.add_nodes_from(self.connections.keys())

        for source, targets in self.connections.items():
            for target in targets:
                G.add_edge(source, target)

        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True)

        for key in self.connections:
            print(key + ': ' + str(self.connections[key]))

        # Draw the graph
        plt.show()

        Neurons = 60000000
        Synapses = 400000000000

        print('')
        print('Ratio of Neurons to Synapses in Thalamus:' + '1:6667')
        print('Number of Neurons in Thalamus:', Neurons)
        print('Number of Synapses in Thalamus:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')
        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 6667
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Thalamus:', synapse_sum)
        metabolism.update_synaptic_sums('Thalamus', synapse_sum)
        return G

    def arousal_of_cortex(self, sleep):
        self.sleep = sleep  # boolean T/F on/off




class Hypothalamus:
    def __init__(self):
        self.feed = False  # brain only takes in glucose
        self.fight = False
        self.flight = False
        self.circadian_Rhythm = None
        self.temperature = None
        self.coitus = None  # Not needed for cyberization of brain
        self.stimulus_threshold = .50  # homeostasis - normal bodily regulation

    def print_info(self, metabolism):
        print("")
        print("Feed:", self.feed)
        print("Fight:", self.fight)
        print("Flight:", self.flight)
        print("Circadian_Rhythm:", self.circadian_Rhythm)
        print("Temperature", self.temperature)
        print("")

        print('Hypothalamus Connectivity Graph:')
        print('')

        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()

        G.add_nodes_from(['Hypothalamus', 'Thalamus', 'Amygdala', 'Hippocampus', 'Frontal Lobe',
                          'Brainstem'])

        G.add_edge('Thalamus', 'Hypothalamus')
        G.add_edge('Hypothalamus', 'Amygdala')
        G.add_edge('Hypothalamus', 'Hippocampus')
        G.add_edge('Hypothalamus', 'Frontal Lobe')
        G.add_edge('Hypothalamus', 'Brainstem')

        for key in G:
            print(key + ': ' + str(G[key]))

        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True)

        plt.show()

        Neurons = 80000000
        Synapses = 800000000

        print('')
        print('Ratio of Neurons to Synapses in Hypothalamus:' + '1:10')
        print('Number of Neurons in Hypothalamus:', Neurons)
        print('Number of Synapses in Hypothalamus:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 10
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Hypothalamus:', synapse_sum)
        metabolism.update_synaptic_sums('Hypothalamus', synapse_sum)
        return G

    def oversee_4_Fs_BodyTemp_Circadian_Rhythm(self, feed, fight, flight, temperature, circadian_Rhythm):
        self.feed = feed  # Boolean True/False
        self.fight = fight  # Boolean True/False
        self.flight = flight  # Boolean True/False
        self.temperature = temperature  # integer value for degrees celsius or farenheit
        self.circadian_Rhythm = circadian_Rhythm  # takes in an integer for hourly intervals #Superchaismatic Nucleus
        # controller

    def dopamine_production_reward(self):
        dopamine = {'dopamine': 0}
        if self.feed:
            dopamine['dopamine'] += 1
        if self.flight:
            dopamine['dopamine'] -= 1
        if self.fight:
            dopamine['dopamine'] += 1
        print('Your dopamine production level:', dopamine)
        return dopamine

    def control_autonomic_nervous_system(self):
        if self.stimulus_threshold > .50:  # stress response
            medulla = medulla_oblongata()
            medulla.set_vitals(120, 80, 16, "normal")
            medulla.set_reflexes(True, False, False)
            medulla.print_info()

    def control_endocrine_systems(self, increase_response):
        endocrine_homeostasis = 50
        hormone_homeostasis = {"testosterone": 50, "estrogen": 50, "oxytocin": 50, "insulin": 50,
                               "cholecystokinin": 50, "cortisol": 50, "adrenaline": 50}
        magnitude = increase_response - endocrine_homeostasis
        while magnitude > 0:
            magnitude -= 1
            for hormone in hormone_homeostasis:
                hormone_homeostasis[hormone] -= 1
            print("Hormone Signaling Pathways Active", hormone_homeostasis)
        return hormone_homeostasis




class LimbicSystem:
    def __init__(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        print('Limbic System Connectivity Graph:')
        print('')

        connections = {
            'Frontal Lobe': ['Amygdala', 'Hippocampus', 'ACC', 'Olfactory Bulb'],
            'Amygdala': ['Frontal Lobe', 'Hippocampus', 'Thalamus', 'Brainstem', 'Olfactory Bulb'],
            'Hippocampus': ['Frontal Lobe', 'Amygdala', 'Thalamus'],
            'ACC-Anterior Cingulate Cortex': ['Frontal Lobe', 'Thalamus'],
            'Thalamus': ['Amygdala', 'Hippocampus', 'ACC', 'Brainstem', 'Olfactory Bulb'],
            'Brainstem': ['Amygdala', 'Thalamus'],
            'Olfactory Bulb': ['Amygdala', 'Thalamus']
        }

        G = nx.DiGraph()

        G.add_nodes_from(connections.keys())

        for source, targets in connections.items():
            for target in targets:
                G.add_edge(source, target)

        for key in G:
            print(key + ': ' + str(G[key]))

        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True)

        plt.show()

        Neurons = 600000
        Synapses = 7000000000

        print('')
        print('Ratio of Neurons to Synapses in Limbic System:' + '1:10000')
        print('Number of Neurons in Limbic System:', Neurons)
        print('Number of Synapses in Limbic System:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 10000
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Limbic System:', synapse_sum)

    def hippocampus(self):
        from BST import BST
        import time
        bst = BST()
        bst.insert(5)
        bst.insert(3)
        bst.insert(7)
        bst.print_tree()

        count = 0
        while True:
            consciousness_input = input(
                "Conscious long-term/short-term memory retrieval in use... create a tree of selectable memories")
            if consciousness_input == "quit":
                break
            if not consciousness_input.isdigit():
                print("Error in decision making.")
            else:
                count += 1
                if count in range(1, 4):
                    bst.insert(int(consciousness_input))
                    bst.print_tree()

        spatialmappingmatrix = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]
        import queue
        queue = queue.Queue()
        row = len(spatialmappingmatrix)
        col = len(spatialmappingmatrix[0])
        result = [
            [0., 0., 0],
            [0., 0., 0],
            [0., 0., 0]
        ]
        for i in range(row):
            for j in range(col):
                if spatialmappingmatrix[i][j] == 0:
                    result[i][j] = 0
                    queue.put((i, j))
                else:
                    result[i][j] = float('inf')
        while not queue.empty():
            curr = queue.get()
            new_row = curr[0]
            new_col = curr[1]

            if (new_row - 1 >= 0 and new_row - 1 < len(result) and new_col >= 0 and new_col < len(result[0])):
                if (result[new_row - 1][new_col] > result[new_row][new_col] + 1):
                    result[new_row - 1][new_col] = result[new_row][new_col] + 1
                    queue.put((new_row - 1, new_col))

            if (new_row + 1 >= 0 and new_row + 1 < len(result) and new_col >= 0 and new_col < len(result[0])):
                if (result[new_row + 1][new_col] > result[new_row][new_col] + 1):
                    result[new_row + 1][new_col] = result[new_row][new_col] + 1
                    queue.put((new_row + 1, new_col))

            if (new_row >= 0 and new_row < len(result) and new_col - 1 >= 0 and new_col - 1 < len(result[0])):
                if (result[new_row][new_col - 1] > result[new_row][new_col] + 1):
                    result[new_row][new_col - 1] = result[new_row][new_col] + 1
                    queue.put((new_row, new_col - 1))

            if (new_row >= 0 and new_row < len(result) and new_col + 1 >= 0 and new_col + 1 < len(result[0])):
                if (result[new_row][new_col + 1] > result[new_row][new_col] + 1):
                    result[new_row][new_col + 1] = result[new_row][new_col] + 1
                    queue.put((new_row, new_col + 1))


        print('')
        print("spatialmappingmatrix:", spatialmappingmatrix, result)
        return bst.sum_nodes()


    def print_hippocampus_graph(self, metabolism):
        print('Hippocampus Connectivity Graph:')
        print('')

        connections = {
            'Hippocampus': ['Frontal Lobe', 'Amygdala', 'Thalamus'],
            'Amygdala': ['Hippocampus', 'Frontal Lobe', 'Thalamus'],
            'Frontal Lobe': ['Hippocampus', 'Amygdala', 'Thalamus', 'Olfactory cortex'],
            'Thalamus': ['Hippocampus', 'Amygdala', 'Frontal Lobe', 'Cingulate gyrus'],
            'Olfactory cortex': ['Frontal Lobe'],
            'Cingulate gyrus': ['Thalamus']
        }

        G = nx.DiGraph()

        G.add_nodes_from(connections.keys())

        for source, targets in connections.items():
            for target in targets:
                G.add_edge(source, target)

        for key in G:
            print(key + ': ' + str(G[key]))

        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True)

        plt.show()

        Neurons = 100000
        Synapses = 1000000000

        print('')
        print('Ratio of Neurons to Synapses in Hippocampus:' + '1:10000')
        print('Number of Neurons in Hippocampus:', Neurons)
        print('Number of Synapses in Hippocampus:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 10000
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Hippocampus:', synapse_sum)
        metabolism.update_synaptic_sums('Hippocampus', synapse_sum)
        return G

    def amygdala(self, metabolism):
        import pandas as pd
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        import warnings
        import random
        from tqdm.notebook import tqdm
        warnings.filterwarnings('ignore')


        import tensorflow as tf
        from tensorflow.keras.utils import to_categorical
        # from keras.preprocessing.image import load_img
        from tensorflow.keras.preprocessing.image import load_img
        from keras.models import Sequential
        from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
        import matplotlib.pyplot as plt
        from PIL import Image
        print("Detecting Emotional Expression...")
        # Specify the directories for training and testing
        TRAIN_DIR = ''
        TEST_DIR = ''

        # Load the dataset
        def load_dataset(directory):
            image_paths = []
            labels = []

            for label in os.listdir(directory):
                label_dir = os.path.join(directory, label)
                if os.path.isdir(label_dir):
                    for filename in os.listdir(label_dir):
                        image_path = os.path.join(label_dir, filename)
                        image_paths.append(image_path)
                        labels.append(label)

                    print(label, "completed")

            return image_paths, labels

        # Convert the dataset into dataframes
        train = pd.DataFrame()
        train['image'], train['label'] = load_dataset(TRAIN_DIR)
        train = train.sample(frac=1).reset_index(drop=True)

        test = pd.DataFrame()
        test['image'], test['label'] = load_dataset(TEST_DIR)

        # Display sample images
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        axes = axes.ravel()

        for i in range(25):
            img = Image.open(train['image'][i])
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(train['label'][i])
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

        # Extract features from images
        def extract_features(images):
            features = []
            for image in tqdm(images):
                img = Image.open(image).convert('L')
                img = np.array(img)
                features.append(img)
            features = np.array(features)
            features = features.reshape(len(features), 48, 48, 1)
            return features

        train_features = extract_features(train['image'])
        test_features = extract_features(test['image'])

        # Normalize the image features
        x_train = train_features / 255.0
        x_test = test_features / 255.0

        # Convert labels to integers
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train['label'])
        y_test = label_encoder.transform(test['label'])

        # Convert labels to categorical
        num_classes = len(label_encoder.classes_)
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

        # Build the model
        model = Sequential()
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(x_train, y_train, batch_size=128, epochs=1, validation_data=(x_test, y_test))

        # Plot accuracy graph
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Graph')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # Plot loss graph
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Graph')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Select a random image index
        image_index = random.randint(0, len(test))

        # Make predictions on a test image
        predicted_probabilities = model.predict(x_test[image_index].reshape(1, 48, 48, 1))
        predicted_label = label_encoder.inverse_transform([predicted_probabilities.argmax()])[0]

        original_label = test['label'][image_index]

        # Display the image and predicted label
        plt.imshow(x_test[image_index].reshape(48, 48), cmap='gray')
        plt.title(f'Original Output: {original_label}\nPredicted Output: {predicted_label}')
        plt.show()

        print('Simulating classical conditioning of fear...')
        import random

        fear = set()
        fear_memory_intensity = 0
        seen_stimuli = {'human', 'ant', 'crow', 'hummingbird', 'human', 'human'}

        for i in range(2):
            for j in range(len(seen_stimuli)):
                random_string = random.choice(list(seen_stimuli))
                print(random_string)
                if random_string == 'human':
                    fear.add('human')
                    fear_memory_intensity += 1
                    print("fear association with stimuli intensity increasing ")
                else:
                    fear_memory_intensity -= 1
                    print("fear association with stimuli intensity decreasing ")
        print('Fear Memory Set:', fear, 'Intensity of Fear', fear_memory_intensity)





        import networkx as nx
        import matplotlib.pyplot as plt

        print('Amygdala Connectivity Graph:')
        print('')

        connections = {
            'Frontal Lobe': ['Amygdala', 'Hippocampus', 'Cingulate Cortex'],
            'Amygdala': ['Frontal Lobe', 'Hippocampus', 'Thalamus', 'Brainstem'],
            'Hippocampus': ['Frontal Lobe', 'Amygdala', 'Thalamus', 'Cingulate cortex'],
            'Cingulate cortex': ['Frontal Lobe', 'Thalamus'],
            'Thalamus': ['Amygdala', 'Hippocampus', 'Cingulate Cortex', 'Brainstem', 'Olfactory bulb'],
            'Brainstem': ['Amygdala', 'Thalamus']
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

        Neurons = 20000000
        Synapses =200000000

        print('')
        print('Ratio of Neurons to Synapses in Amygdala:' + '1:10')
        print('Number of Neurons in Amygdala:', Neurons)
        print('Number of Synapses in Amygdala:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 10
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Amygdala:', synapse_sum)
        metabolism.update_synaptic_sums('Amygdala', synapse_sum)
        return G

    def cingulate_gyrus(self,  metabolism):
        import random
        stimulusintensity = random.randint(0, 1)
        print(stimulusintensity)

        behavior_likelihood = 0
        habituation = 0
        sensitization = 0

        # neurotransmitters involved in +/- learning&&memory
        neurotransmitters = {
            'glutamate': 0,
            'GABA': 0,
            'dopamine': 0,
            'serotonin': 0
        }

        if stimulusintensity == 1:
            behavior_likelihood += 1
            sensitization += 1
            for i in neurotransmitters:
                neurotransmitters[i] += 1

        if stimulusintensity == 0:
            behavior_likelihood -= 1
            habituation -= 1
            for j in neurotransmitters:
                neurotransmitters[j] -= 1

        if habituation == -1:
            print('Low-Intensity Simuli - Decreasing Behavior and Memory...', 'Neurotransmitter Status:',
                  neurotransmitters)
        else:
            print('High-Intensity Simuli - Increasing Behavior and Memory...', 'Neurotransmitter Status:',
                  neurotransmitters)

        import networkx as nx
        import matplotlib.pyplot as plt

        print('Cingulate Gyrus/Cortex Connectivity Graph:')
        print('')

        connections = {
            'Cingulate Cortex': ['Frontal Lobe', 'Thalamus', 'Amygdala'],
            'Thalamus': ['Cingulate Cortex', 'Amygdala', 'Brainstem'],
            'Amygdala': ['Cingulate Cortex', 'Thalamus', 'Brainstem'],
            'Hippocampus': ['Thalamus', 'Brainstem'],
            'Frontal Lobe': ['Cingulate Cortex'],
            'Brainstem': [ 'Thalamus', 'Amygdala', 'Hippocampus']
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

        Neurons = 10000000
        Synapses = 150000000

        print('')
        print('Ratio of Neurons to Synapses in Cingulate Gyrus:' + '1:15')
        print('Number of Neurons in Cingulate Gyrus:', Neurons)
        print('Number of Synapses in Cingulate Gyrus:', Synapses)
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
        print('Sum of synaptic weights in Cingulate Gyrus:', synapse_sum)
        metabolism.update_synaptic_sums('Cingulate Gyrus', synapse_sum)
        return G

    def olfactory_bulb(self, metabolism):
        import random
        smell = random.randint(0, 1)
        if smell == 1:
            print("Smell Receptor Activated")
            from OccipitalLobe import OccipitalLobe
            olfactoryReception = OccipitalLobe()
            print('Olfaction Receptivity Matrix:', olfactoryReception.detailPixel())

        import networkx as nx
        import matplotlib.pyplot as plt

        print('Olfactory Connectivity Graph:')
        print('')

        connections = {
            'Olfactory Bulb': [ 'Amygdala',  'Thalamus'],
            'Amygdala': ['Olfactory Bulb'],
            'Thalamus': ['Olfactory Bulb']
        }

        G = nx.DiGraph()

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

        Neurons = 500000
        Synapses = 5000000000

        print('')
        print('Ratio of Neurons to Synapses in Olfactory Bulb:' + '1:10000')
        print('Number of Neurons in Olfactory Bulb:', Neurons)
        print('Number of Synapses in Olfactory Bulb:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 10000
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Olfactory Bulb:', synapse_sum)
        metabolism.update_synaptic_sums('Olfactory Bulb', synapse_sum)
        return G

class BasalGanglia:
    def __init__(self):
        self.synapse_sum = 0;
        print('')

    def control_movement_task_setting(self):
        # initialize the movement counters
        up_count = 0
        down_count = 0
        left_count = 0
        right_count = 0
        rotation_value = None

        # keep looping until the user types 'quit'
        while True:
            # get user input
            user_input = input("Enter a direction (Up, Down, Left, Right), or type 'Rotate' to rotate: ")

            # process user input
            if user_input == 'quit':
                break
            elif user_input == 'Up':
                up_count += 1
            elif user_input == 'Down':
                down_count += 1
            elif user_input == 'Left':
                left_count += 1
            elif user_input == 'Right':
                right_count += 1
            elif user_input == 'Rotate':
                rotation_value = BasalGanglia.rotate()

            else:
                print(
                    "Invalid input. Please enter a valid direction (Up, Down, Left, Right) or type 'Rotate' to rotate.")
                continue

        # print the final movement counts
        print(f"Total movements: Up - {up_count}, Down - {down_count}, Left - {left_count}, Right - {right_count}, "
              f"Rotate - {rotation_value}")

    @staticmethod
    def rotate() -> float:
        trigValue = input("Enter θ")
        while not trigValue.isdigit():
            trigValue = input("Invalid input. Please enter theta, a numerical angle value")
        theta = np.radians(float(trigValue))
        c, s = np.cos(theta), np.sin(theta)
        rotationValue = np.array(((c, -s), (s, c)))
        print(f"Theta is: {trigValue} Rotation Value is: {rotationValue}")
        return rotationValue

    def print_info(self, metabolism):
        print('Basal Ganglia Connectivity Graph:')
        print('')

        connections = {
            'Basal Ganglia': ['Cortex', 'Thalamus'],
            'Cortex': ['Basal Ganglia'],
            'Thalamus': ['Basal Ganglia']
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

        Neurons = 1.6 * 10 ** 7
        Synapses = 1.6 * 10 ** 11

        print('')
        print('Ratio of Neurons to Synapses in Basal Ganglia:' + '1:10000')
        print('Number of Neurons in Basal Ganglia:', Neurons)
        print('Number of Synapses in Basal Ganglia:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 10000
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Basal Ganglia:', synapse_sum)
        metabolism.update_synaptic_sums('Basal Ganglia', synapse_sum)
        return G

class BasalForebrain:
    def __init__(self):
        self.sleep = None;
        self.wakefulness_homeostasis = 0.5
        self.stimulus = random.randint(0, 1)

    def arousal_of_cortex(self, sleep, stimulus, metabolism):
        self.sleep = sleep
        self.stimulus = stimulus
        neurotransmitters = {'acetylcholine': 0, 'GABA': 0}

        if stimulus == 1:
            for i in neurotransmitters:
                neurotransmitters[i] += 1
                print('arousal initiation beginning...')
        else:
            for j in neurotransmitters:
                neurotransmitters[j] -= 1
                print('de-arousal initiation beginning...')

        print('Basal Forebrain Connectivity Graph:')
        print('')

        connections = {
            'Basal Forebrain': ['Cortex', 'Hippocampus', 'Amygdala'],
            'Cortex': ['Basal Forebrain'],
            'Hippocampus': ['Basal Forebrain'],
            'Amygdala': ['Basal Forebrain']
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

        Neurons = 1000000
        Synapses = 10000000000

        print('')
        print('Ratio of Neurons to Synapses in Basal Forebrain:' + '1:10000')
        print('Number of Neurons in Basal Forebrain:', Neurons)
        print('Number of Synapses in Basal Forebrain:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 10000
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Basal Forebrain:', synapse_sum)
        metabolism.update_synaptic_sums('Basal Forebrain', synapse_sum)
        return G
