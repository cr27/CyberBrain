from Forebrain import Forebrain, LimbicSystem, Hypothalamus, Thalamus, BasalGanglia, BasalForebrain
# from OccipitalLobe import OccipitalLobe
# from TemporalLobe import TemporalLobe
# from ParietalLobe import ParietalLobe
# from FrontalLobe import FrontalLobe
# from binarytree import Node, print_tree_preorder
# from Hindbrain import medulla_oblongata, Pons, cerebellum
# from Midbrain import Tectum, Tegmentum
# from Cortical_Stack import Cortical_Stack,Neurome
# from BrainNetwork import BrainNetwork
from Metabolism import Metabolism
######################### Metabolism Initialization ###########################
metabolism = Metabolism()
######################################### NanoNeuroInterface ##################
# - integrating chemical based nanobot payload with neuron neurotransmitters for chemical synapses
# import numpy as np
# import nanoneuro_interface
# from Signal import Signal
#
# # Create an instance of the Signal class
# signal = Signal()
#
# # Generate the signal and extract the amplitude
# signal.generate_signal()
#
# # Set the amplitude
# signal.set_amplitude()
#
# # Get the amplitude
# amplitude = signal.get_amplitude()
#
# # nanobot = nanoneuro_interface.recieveSignal(amplitude)
#
# #Debug/Test
# nanobot = nanoneuro_interface.recieveSignal(0.4)
# #Debug/Test
# #nanobot2 = nanoneuro_interface.recieveSignal(0.5)
signal = Signal()
signal.generate_language_memory_injection_signal()
signal.generate_bicycle_riding_memory_injection_signal()
signal.generate_GPS_memory_injection_signal()
########################################## CorticalStack - Connects PNS and Neurome to CNS
# Cortical_Stack = Cortical_Stack()
# Cortical_Stack.Peripheral_Nervous_System()
# Neurome = Neurome();
# Neurome.Immune_System()
# Neurome.Excretory_System()
# Neurome.Circulatory_System()
# Neurome.Respiratory_System()
# Neurome.Digestive_System()
# Neurome.Integumentary_System()
# Neurome.Endocrine_System()
# Neurome.Musculoskeletal_System()
# Neurome.Neurome_print_connections()
######################################### Forebrain
Forebrain = Forebrain()
# Thalamus = Thalamus()
# Thalamus.intrinsic_information_processing()
# Thalamus.print_info(metabolism)
#
# Hypothalamus = Hypothalamus()
# Hypothalamus.oversee_4_Fs_BodyTemp_Circadian_Rhythm(True,True,True,70,5) #superchiasmatic nucleus - bodys internal clock
# Hypothalamus.dopamine_production_reward()
# Hypothalamus.control_endocrine_systems(52) #value must be greater than 51
# Hypothalamus.control_autonomic_nervous_system()
# Hypothalamus.print_info(metabolism)
#
LimbicSystem = LimbicSystem()
LimbicSystem.hippocampus(metabolism)
# LimbicSystem.amygdala(metabolism)
# LimbicSystem.cingulate_gyrus(metabolism)
# LimbicSystem.olfactory_bulb(metabolism)
#
# BasalGanglia = BasalGanglia()
# BasalGanglia.control_movement_task_setting() #**
# BasalGanglia.print_info(metabolism)
#
# BasalForebrain  = BasalForebrain()
# BasalForebrain.arousal_of_cortex(False, 1, metabolism)
########################################## MidBrain
# Tectum = Tectum()
# Tectum.Superior_Colliculus()
# Tectum.inferior_colliculus()
# Tectum.print_info(metabolism)
#
# Tegmentum = Tegmentum()
# Tegmentum.print_info(metabolism)
# ######################################### Hindbrain
# medulla = medulla_oblongata()
# medulla.set_vitals(120, 80, 16, "normal")
# medulla.set_reflexes(True, False, False)
# medulla.print_info(metabolism)
#
# pons = Pons()
# pons.set_sleep(True)
# pons.set_respiration(60)
# pons.set_facialmovement(True)
# pons.print_info(metabolism)
#
# cerebellum = cerebellum()
# cerebellum.set_balance(True)
# cerebellum.set_posture(True)
# cerebellum.set_motorcoordination(True)
# cerebellum.set_coordinatemovement(True)
# cerebellum.reflexes()
# cerebellum.print_info(metabolism)
################################ Frontal Lobe
# FrontalLobe = FrontalLobe()
# FrontalLobe.print_info(metabolism)
# FrontalLobe.delayedGratification()
# FrontalLobe.selfControl()
# FrontalLobe.planning()
# FrontalLobe.cultural_rules()
# FrontalLobe.emotionalExpression()
# FrontalLobe.brocasArea()
# FrontalLobe.M1()
# ############################### Occiptial Lobe
# OccipitalLobe = OccipitalLobe()
# OccipitalLobe.colorPixel()
# OccipitalLobe.detailPixel()
# OccipitalLobe.depthPixel()
# OccipitalLobe.motionPixel()
# OccipitalLobe.print_info(metabolism)
# ############################# Parietal Lobe
# ParietalLobe = ParietalLobe()
# ParietalLobe.print_info(metabolism)
# ParietalLobe.mirrorNeuronSystem()
# ParietalLobe.penfieldMap()
# ParietalLobe.S1()
# ########################## Temporal Lobe
# TemporalLobe = TemporalLobe()
# TemporalLobe.print_info(metabolism)
# TemporalLobe.whowhatpathway()
# TemporalLobe.A1()
############################# Combination of Connection Graph Brain Regions Working Together
# BrainNetwork = BrainNetwork();
# BrainNetwork.combine_connectivity_graphs()
# BrainNetwork.print_brain_network()
# BrainNetwork.simulate_brain_network() # contains metabolism.print_info
###############################








