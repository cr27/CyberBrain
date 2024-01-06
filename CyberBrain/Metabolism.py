# In the context of artificial neural networks, the concept of "neural metabolism" is often used metaphorically to
# describe the computational resources or computational cost associated with the operations performed by artificial
# neurons and synapses. These computational costs can include the calculations
# involved in processing input signals, updating synaptic weights, and performing network operations.
#########################################################################################################################
#Dictionary for Brain Region and its respective Synaptic weight
class Metabolism:
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.synaptic_sums = {
        }

    def update_synaptic_sums(self, region, sum_value):
        self.synaptic_sums[region] = sum_value

    def synaptic_sum(self):
        length = len(self.synaptic_sums)
        return length

    def print_info(self):
        while self.synaptic_sums:
            max_sum = max(self.synaptic_sums.values())
            max_sum_regions = [region for region, sum_value in self.synaptic_sums.items() if sum_value == max_sum]
            print(" ")
            print('Brain Regions with the current highest metabolic demand:')
            for region in max_sum_regions:
                print(region + ':', self.synaptic_sums[region])

            for region in max_sum_regions:
                del self.synaptic_sums[region]

            print('Remaining brain regions and their associated sums:')
            for region, sum_value in self.synaptic_sums.items():
                print(region + ':', sum_value)

            print('')

    # def print_info(self):
    #     while self.synaptic_sums:
    #         max_sum = max(self.synaptic_sums.values())
    #         max_sum_regions = [region for region, sum_value in self.synaptic_sums.items() if sum_value == max_sum]
    #
    #         print('Brain Regions with the current highest metabolic demand:', max_sum_regions)
    #
    #         for region in max_sum_regions:
    #             del self.synaptic_sums[region]
    #
    #         print('Remaining brain regions and their associated sums:', self.synaptic_sums)
    #         print('')















