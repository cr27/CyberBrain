import networkx as nx
from matplotlib import pyplot as plt, image as mpimg
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization
import kapre
from tensorflow.keras.models import Sequential
from kapre.composed import get_melspectrogram_layer, get_log_frequency_spectrogram_layer
import numpy as np
import librosa
import matplotlib.pyplot as plt
import numpy as np
from CMOS_Neuron_Memresistor_Synapse import CMOSNeuron, MemristorSynapse
from Metabolism import Metabolism
# : Higher Visual, Audition, Emotion & Language Comprehension

#The lateral surface of the Cerebral Cortex – the temporal lobe - is responsible for higher level
# processing of various sensory modalities and cognitive functions. This includes higher level visual
# processing in the Inferior and Medial Temporal regions, where the "Who/What" pathway and "Where/How"
# pathway are located respectively. The Inferior Temporal region also includes specialized cells for
# facial recognition.The lateral surface also includes the primary auditory projection area (A1) from the
# Medial Geniculate Nucleus of the Thalamus, as well as higher auditory areas involved in language comprehension,
# such as Wernicke's Area located in the left hemisphere. Additionally, the Anterior Temporal regionis
# implicated in emotional expression and interpretation, particularly in the right hemisphere.

#Medial Temporal (MT), part of other main visual pathway (to Parietal), the “Where/How” pathway
#-Includes many Motion Sensitive cells, including Optic Flow detectors
#facial/emotional recognition, audio recognition

class TemporalLobe():
    def __init__(self):
        print("")
    def print_info(self, metabolism):
        print("Temporal Lobe Connection Graph:")
        print(" ")
        connections = {
            'Temporal Lobe': ['Cingulate Cortex', 'Frontal Lobe', 'Amygdala', 'Hippocampus'],
            'Cingulate Cortex': ['Temporal Lobe', 'Frontal Lobe', 'Amygdala', 'Thalamus'],
            'Frontal Lobe': ['Temporal Lobe', 'Cingulate Cortex', 'Amygdala'],
            'Amygdala': ['Temporal Lobe', 'Cingulate Cortex', 'Frontal Lobe', 'Hippocampus',
                         'Thalamus'],
            'Hippocampus': ['Temporal Lobe', 'Amygdala', 'Thalamus'],
            'Thalamus': ['Cingulate Cortex', 'Amygdala', 'Hippocampus']
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

        Neurons = 150000000
        Synapses = 5000000000

        print(" ")
        print('Ratio of Neurons to Synapses in Temporal Lobe:' + '1:33')
        print('Number of Neurons in Temporal Lobe:', Neurons)
        print('Number of Synapses in Temporal Lobe:', Synapses)
        print('Activating CMOS and Memresistor based Electrical Synapses and Neurons...')

        neuron = CMOSNeuron(Vth=0.5)
        synapse = MemristorSynapse(alpha=0.1)
        num_synapses = 33
        num_neurons = 1
        Vin = np.random.rand()
        Vout = neuron.process(Vin)
        synapse_sum = 0
        for i in range(num_neurons):
            for j in range(num_synapses):
                synapse_sum += synapse.process(Vin)
        print('Sum of synaptic weights in Temporal Lobe:', synapse_sum)
        metabolism.update_synaptic_sums('Temporal Lobe', synapse_sum)
        return G

    def wernickesArea(self):
        from nltk.tokenize import word_tokenize
        from nltk.text import Text
        my_string = "Two plus two is four, minus one that's three — quick maths. Every day man's on the block. " \
                    "Smoke trees. See your girl in the park, that girl is an uckers. " \
                    "When the thing went quack quack quack, your men were ducking! Hold tight Asznee, " \
                    "my brother. He's got a pumpy. Hold tight my man, my guy. He's got a frisbee. I trap, trap, " \
                    "trap on the phone. Moving that cornflakes, rice crispies. Hold tight my girl Whitney."
        tokens = word_tokenize(my_string)
        tokens1 = [word.lower() for word in tokens]
        tokens2 = tokens[:5]
        return tokens, tokens1, tokens2

    def A1(X):
        src, sr = librosa.load(
        '', sr=None,
        mono=True)
        print('Audio length: %d samples, %04.2f seconds. \n' % (len(src), len(src) / sr) +
              'Audio sample rate: %d Hz' % sr)
        dt = 1.0
        _src = src[:int(sr * dt)]
        src = np.expand_dims(_src, axis=1)
        input_shape = src.shape
        print(input_shape)

        melgram = get_melspectrogram_layer(input_shape=input_shape,
                                           n_mels=128,
                                           mel_norm='slaney',
                                           pad_end=True,
                                           n_fft=512,
                                           win_length=400,
                                           hop_length=160,
                                           sample_rate=sr,
                                           db_ref_value=1.0,
                                           return_decibel=True,
                                           input_data_format='channels_last',
                                           output_data_format='channels_last')
        norm = LayerNormalization(axis=2)
        melgram.shape = (16000, 1)
        model = Sequential()
        model.add(melgram)
        model.add(norm)
        model.summary()

        batch = np.expand_dims(src, axis=0)
        X = model.predict(batch).squeeze().T
        # A1(X)

        plt.title('Normalized Frequency Histogram')
        plt.hist(X.flatten(), bins='auto')
        plt.show()
        print(" ")
        print("A1 Connection Graph:")
        print(" ")
        connections = {
            'Thalamus': ['A1'],
            'A1': ['Thalamus']
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

        neurons = 1000000
        synapses = 10000000

        print(" ")
        print('Ratio of Neurons to Synapses in A1:' + '1:10')
        print('Number of Neurons in A1:', neurons)
        print('Number of Synapses in A1:', synapses)
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
        print('Sum of synaptic weights in A1:', synapse_sum)
        return G

    def wherehowpathway(coords):
        # Define a 2D grid based on the user input coordinates
        where = np.zeros((10, 10))
        for coord in coords:
            where[coord[0], coord[1]] = 1

        # Simulate changes over time for the how pathway
        how = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                # Here, you could modify this calculation to simulate different types of changes
                how[i, j] = np.sin(i + j)

        return where, how

    # Example usage:
    coords = [(2, 4), (5, 7), (8, 1)]
    where, how = wherehowpathway(coords)
    print("Where pathway:\n", where)
    print("How pathway:\n", how)

    #Inferior Temporal (IT) includes higher Visual area, along “Who/What” pathway, including Face Cells
    def whowhatpathway(self):
        import os
        import numpy as np
        from PIL import Image, ImageOps
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.model_selection import train_test_split
        import glob

        # Define the features and labels for the model
        animal_labels = ["n02085620", "n02085782"]
        Chihuahua_dir = ""
        Japanese_spaniel_dir = ""

        yes_images = []
        no_images = []

        yes = []
        no = []

        for filename in glob.iglob(Chihuahua_dir + '**/*', recursive=True):
            im = Image.open(filename)
            im = im.resize((128, 128))
            im = np.array(im)
            im = np.expand_dims(im, axis=-1)  # Add an extra dimension for grayscale
            yes_images.append(im)

        for filename in glob.iglob(Japanese_spaniel_dir + '**/*', recursive=True):
            im = Image.open(filename)
            im = im.resize((128, 128))
            im = np.array(im)
            im = np.expand_dims(im, axis=-1)  # Add an extra dimension for grayscale
            no_images.append(im)

        for image in yes_images:
            yes.append(image)

        for image in no_images:
            no.append(image)

        # Split data into training and testing sets
        X = yes + no
        yes_label = np.ones(len(yes))
        no_label = np.zeros(len(no))
        y = np.concatenate((yes_label, no_label), axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        X_train = np.asarray(X_train).reshape(-1, 128, 128, 1)
        X_test = np.asarray(X_test).reshape(-1, 128, 128, 1)

        # Determine the number of input features
        num_features = (128, 128, 1)

        # Define the TensorFlow model
        model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=num_features),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the TensorFlow model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the TensorFlow model
        model.fit(X_train, y_train, epochs=10)

        # Evaluate the model on the testing set
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print("Test accuracy:", test_acc)

        # Load a single image for prediction
        img = Image.open("")
        img = img.resize((128, 128))
        img = np.array(img)
        img = np.expand_dims(img, axis=-1)  # Add an extra dimension for grayscale
        img = img.reshape(1, 128, 128, 1)

        # Feed image into TensorFlow model for analysis
        prediction = model.predict(img)
        predicted_label = animal_labels[int(prediction[0][0])]
        print("Predicted label:", predicted_label)
        plt.title("Dog Image")
        plt.xlabel("X pixel scaling")
        plt.ylabel("Y pixels scaling")

        image = mpimg.imread("")
        plt.imshow(image)
        plt.show()




def fusiformGyrus():
     print("hi")



