class Signal():
    def __init__(self):
        self.amplitude = 0.0
        self.frequency = 0.0

        Frequency_integer_code_find = None  # turn into global variable - .self

    def generate_signal(self):
        import random
        self.amplitude = random.uniform(0.3, 0.9)
        print(self.amplitude)

    def set_amplitude(self):
        # self.amplitude = np.max(self.envelope)
        print("Your amplitude is:", self.amplitude)

    def get_amplitude(self):
        return self.amplitude

    def brain_region_signals(self):
        import random
        Frequency_integer_code_find = None
        brain_region_signal_dict = {'Frontal Lobe': 1, 'Parietal Lobe': 2, 'Temporal Lobe': 3, 'Occipital Lobe': 4,
                                    'Medulla Oblangata': 5, 'Pons': 6, 'Cerebellum': 7, 'Tectum': 8, 'Tegmentum': 9,
                                    'Thalamus': 10, 'Hypothalamus': 11, 'Hippocampus': 12, 'Amygdala': 13,
                                    'Cingulate Gyrus': 14, 'Basal Ganglia': 15, 'Basal Forebrain': 16,
                                    'Olfactory Bulb': 17}
        print(" ")
        for key, value in brain_region_signal_dict.items():
            print("Available Brain Region's and Complementary Brain Region Frequency integer code:", key, value)

        keyword = 'found'
        corresponding_key = None

        while Frequency_integer_code_find != keyword:
            print('')
            print('Ramping up signal generator Frequency integer code...')
            print('')
            print('Generating Frequency integer code for nanoneuralrobots and brain region')
            self.frequency = random.randint(1, 17)

            print('')
            print('The generated Frequency integer code is', self.frequency)
            print('')
            Frequency_integer_code_find = input(
                "Enter 'found' if the brain region Frequency integer code contains the hippocampus or thalamus,"
                " otherwise press enter: ")

            if Frequency_integer_code_find.isdigit():
                Frequency_integer_code_find = int(Frequency_integer_code_find)
                if Frequency_integer_code_find in brain_region_signal_dict.values():
                    corresponding_key = next((key for key, value in brain_region_signal_dict.items()
                                              if value == Frequency_integer_code_find), None)
                    break

        print(" You've found your amplitude! ", " ", "Frequency integer code =", self.frequency)
        print(" ")
        print("Brain Region:", corresponding_key)

    def generate_language_memory_injection_signal(self):

        import random

        Frequency_integer_code_find = None

        language_memory_injection = {'Hello': 'Ni hao', 'Contest': 'bisai'}

        print(" ")
        for key, value in language_memory_injection.items():
            print("Memory Injection for Chinese:", key, value)

        self.brain_region_signals()

        print(" ")
        print("Matching the brain region amplitude with a char list that describes the language and learning/knowledge",
              "you would like to download to your hippocampus & Thalamus")

        char_language_list = []
        keys = language_memory_injection

        for letter in keys:
            char_language_list.extend(list(letter))

        print("")
        print(char_language_list)

        ascii_values = {char: [ord(char)] * char_language_list.count(char) for char in char_language_list}

        print(ascii_values)

        ascii_letter_Frequency_integer_codes = {}

        for value in range(65, 91):  # ASCII values for capital letters (A-Z)
            ascii_letter_Frequency_integer_codes[value] = 0

        for value in range(97, 123):  # ASCII values for lowercase letters (a-z)
            ascii_letter_Frequency_integer_codes[value] = 0

        # Count the frequencies of ASCII values based on the character dictionary
        for value_list in ascii_values.values():
            for value in value_list:
                if value in ascii_letter_Frequency_integer_codes:
                    ascii_letter_Frequency_integer_codes[value] += 1

        # Print the ASCII values and their frequencies for letters (excluding zero frequencies)
        for value, frequency in ascii_letter_Frequency_integer_codes.items():
            if frequency > 0:
                print(
                    f"ASCII value/Amplitude: {value}, Frequency: {frequency}, Brain Region Frequency: "
                    f"{Frequency_integer_code_find}")
        print("")
        print("Signal Frequencies being sent to nanoneuralrobot and brain region from nanoneurointerface")

    def generate_bicycle_riding_memory_injection_signal(self):

        Frequency_integer_code_find = None

        self.brain_region_signals()

        bike_riding_memory_injection = {'balance': 200, 'shift_body_weight': 201, 'pedal_amount': 202,
                                        'press_brake': 203}

        import random

        print(" Procedural Memory is Stored in the Basal Ganglia and Cerebellum")

        print(" ")
        print('Generating Bicycle Ride Obstacles...')

        balance = False
        pedal_amount = 0
        press_brake = False

        print(" ")
        print('Bicycle Ride Learning Simulation commencing...')

        shift_body_weight = random.randint(0, 1)
        obstacle = random.randint(0, 1)
        bike_Riding_Frequency_int_code_list = []

        while (shift_body_weight != 1):

            shift_body_weight = random.randint(0, 1)
            obstacle = random.randint(0, 1)

            if shift_body_weight == obstacle:
                print(" ")
                print('Avoided Obstacle')
                balance = True
                print('Balance:', balance)
                bike_Riding_Frequency_int_code_list.append(balance)
                pedal_amount += 1
                print('Pedal Amount:', pedal_amount)
                bike_Riding_Frequency_int_code_list.append(pedal)
            else:
                print('Obstacles not avoided')
                print(" ")
                print("Balance:", balance)
                pedal_amount += 1
                press_brake = True
                print("Braking...", press_brake)
                bike_Riding_Frequency_int_code_list.append(press_brake)
                print('Pedal Amount:', pedal_amount)

        print("Avoided Obstacle - Bike Riding Memory Complete and Ready for Injection")
        print("Frequencies from nanoneurointerface being sent to nanobot", bike_Riding_Frequency_int_code_list)

        ## GPS Memory is int coded with 300 + point of origin distance sum based integers relative to signal frequencies

    def generate_GPS_memory_injection_signal(self):

        print(" Spatial Memory is Stored in the Hippocampus")

        self.brain_region_signals()

        ## Generate Coordinates and distance from coordinates
        points = [[1, 3], [-2, 2]]
        k = 1
        import queue
        queue = queue.Queue()
        result = []
        GPS_Frequency_int_code_list = []

        distance = 0
        distanceSum = 0
        seenset = {}
        seenset_2D = {}
        dict_points = {}

        for i in range(len(points)):
            queue.put(points[i])

        for i in range(len(points)):
            result.append(queue.get())
            for point in result[i]:
                distance = point ** 2
                distanceSum += distance
                print(distanceSum)
            distance = 0
            GPS_Frequency_int_code = distanceSum + 300
            GPS_Frequency_int_code_list.append(GPS_Frequency_int_code)
            distanceSum = 0

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

        print("Distance from Point of Origin Frequencies", GPS_Frequency_int_code_list)

        result_frequency = 0
        for i in range(len(result)):
            for j in range(len(result[0])):
                result_frequency += result[i][j]

        result_frequency += 300
        GPS_Frequency_int_code_list.append(result_frequency)

        print("")
        print("Spatialmappingmatrix, spacing frequency combination of left, up, right, down from a distance:",
              result_frequency)
        print("Frequencies from nanoneurointerface being sent to nanobot",
              GPS_Frequency_int_code_list)