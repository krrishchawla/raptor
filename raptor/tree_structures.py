from typing import Dict, List, Set


class Metadata:
    def __init__(self, year, company, sector) -> None:
        self.year = year
        self.company = company
        self.sector = sector
    

class Node:
    """
    Represents a node in the hierarchical tree structure.
    """

    def __init__(self, text: str, index: int, children: Set[int], metadata: Metadata, embeddings) -> None:
        self.text = text
        self.index = index
        self.children = children
        self.embeddings = embeddings
        self.metadata = metadata


class Tree:
    """
    Represents the entire hierarchical tree structure.
    """

    def __init__(
        self, all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes
