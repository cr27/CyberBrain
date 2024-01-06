import networkx as nx
from TemporalLobe import TemporalLobe
from ParietalLobe import ParietalLobe
from OccipitalLobe import OccipitalLobe
from FrontalLobe import FrontalLobe
from Hindbrain import medulla_oblongata, Pons, cerebellum
from Midbrain import Tectum, Tegmentum
from Forebrain import Forebrain, LimbicSystem, Hypothalamus, Thalamus, BasalGanglia, BasalForebrain
from matplotlib import pyplot as plt
from Metabolism import Metabolism

metabolism = Metabolism() #initialize metabolism and pass to parameters

class BrainNetwork:
    def __init__(self):
        print(" ")
        self.graph = nx.DiGraph()

    def combine_connectivity_graphs(self):
        connectivity_graph_1 = TemporalLobe()
        connectivity_graph_2 = ParietalLobe()
        connectivity_graph_3 = OccipitalLobe()
        connectivity_graph_4 = FrontalLobe()

        ##Hindbrain
        connectivity_graph_5 = medulla_oblongata()
        connectivity_graph_6 = Pons()
        connectivity_graph_7 = cerebellum()

        ##Midbrain
        connectivity_graph_8 = Tectum()
        connectivity_graph_9 = Tegmentum()

        ##Forebrain
        connectivity_graph_10 = Thalamus()
        connectivity_graph_11 = Hypothalamus()
        connectivity_graph_12 = LimbicSystem()
        connectivity_graph_13 = BasalGanglia()
        connectivity_graph_14 = BasalForebrain()

        ##Temporal Lobe
        graph1 = connectivity_graph_1.print_info(metabolism)
        graph2 = connectivity_graph_1.A1()

        ##Parietal Lobe
        graph3 = connectivity_graph_2.S1()
        graph4 = connectivity_graph_2.print_info(metabolism)

        ##Occipital Lobe
        graph5 = connectivity_graph_3.print_info(metabolism)

        ##Frontal Lobe
        graph6 = connectivity_graph_4.print_info(metabolism)
        graph7 = connectivity_graph_4.M1()

        ##medulla oblangata
        graph8 = connectivity_graph_5.print_info(metabolism)

        ##Pons
        graph9 = connectivity_graph_6.print_info(metabolism)

        ##Cerebellum
        graph10 = connectivity_graph_7.print_info(metabolism)

        ##Tectum
        graph11 = connectivity_graph_8.print_info(metabolism)

        ##Tegmentum
        graph12 = connectivity_graph_9.print_info(metabolism)

        #Thalamus
        graph13 = connectivity_graph_10.print_info(metabolism)

        #Hypothalamus
        graph14 = connectivity_graph_11.print_info(metabolism)#

        #Limbic System
        graph15 = connectivity_graph_12.print_hippocampus_graph(metabolism)
        graph16 = connectivity_graph_12.amygdala(metabolism)
        graph17 = connectivity_graph_12.olfactory_bulb(metabolism)
        graph18 = connectivity_graph_12.cingulate_gyrus(metabolism)

        # Basal Ganglia
        graph19 = connectivity_graph_13.print_info(metabolism)

        # Basal Forebrain
        graph20 = connectivity_graph_14.arousal_of_cortex(False, 1, metabolism)

        # Add nodes and edges from each connectivity graph to the combined brain network graph
        self.graph.add_nodes_from(graph1.nodes)
        self.graph.add_edges_from(graph1.edges)

        self.graph.add_nodes_from(graph2.nodes)
        self.graph.add_edges_from(graph2.edges)

        self.graph.add_nodes_from(graph3.nodes)
        self.graph.add_edges_from(graph3.edges)

        self.graph.add_nodes_from(graph4.nodes)
        self.graph.add_edges_from(graph4.edges)

        self.graph.add_nodes_from(graph5.nodes)
        self.graph.add_edges_from(graph5.edges)

        self.graph.add_nodes_from(graph6.nodes)
        self.graph.add_edges_from(graph6.edges)

        self.graph.add_nodes_from(graph7.nodes)
        self.graph.add_edges_from(graph7.edges)

        self.graph.add_nodes_from(graph8.nodes)
        self.graph.add_edges_from(graph8.edges)

        self.graph.add_nodes_from(graph9.nodes)
        self.graph.add_edges_from(graph9.edges)

        self.graph.add_nodes_from(graph10.nodes)
        self.graph.add_edges_from(graph10.edges)

        self.graph.add_nodes_from(graph11.nodes)
        self.graph.add_edges_from(graph11.edges)

        self.graph.add_nodes_from(graph12.nodes)
        self.graph.add_edges_from(graph12.edges)

        self.graph.add_nodes_from(graph13.nodes)#
        self.graph.add_edges_from(graph13.edges)

        self.graph.add_nodes_from(graph14.nodes)
        self.graph.add_edges_from(graph14.edges)

        self.graph.add_nodes_from(graph15.nodes)
        self.graph.add_edges_from(graph15.edges)

        self.graph.add_nodes_from(graph16.nodes)
        self.graph.add_edges_from(graph16.edges)

        self.graph.add_nodes_from(graph17.nodes)
        self.graph.add_edges_from(graph17.edges)

        self.graph.add_nodes_from(graph18.nodes)
        self.graph.add_edges_from(graph18.edges)

        self.graph.add_nodes_from(graph19.nodes)
        self.graph.add_edges_from(graph19.edges)

        self.graph.add_nodes_from(graph20.nodes)
        self.graph.add_edges_from(graph20.edges)

    def print_brain_network(self):
        print("")
        print("Brain Network:")
        for node, neighbors in self.graph.adjacency():
            print(f"{node}: {neighbors}")

        print(self.graph)
        pos = nx.circular_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True)
        plt.show()

    # Additional processing or analysis of the combined brain network can be performed here
    def simulate_brain_network(self):
        # Perform simulation or access specific points in the brain network
        metabolism.print_info()


