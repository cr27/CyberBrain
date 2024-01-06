import numpy as np

class CMOSNeuron:
    def __init__(self, Vth):
        self.Vth = Vth

    def process(self, Vin):
        Vout = 1 if Vin >= self.Vth else 0
        return Vout, Vin

class MemristorSynapse:
    def __init__(self, alpha):
        self.W = 0
        self.alpha = alpha

    def process(self, Vin):
        self.W = self.W + self.alpha * Vin
        return self.W

neuron = CMOSNeuron(Vth=0.5)
synapse = MemristorSynapse(alpha=0.1)

Vin = np.random.rand()
Vout = neuron.process(Vin)
W = synapse.process(Vin)

print("Artificial Neural Voltage Input:", Vin)
print("Artificial Neural Voltage Output:", Vout)
print("Artificial Neural Synaptic Weight:", W)

