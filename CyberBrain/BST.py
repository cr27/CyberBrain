class BST:
    class Node:
        def __init__(self, value):
            self.value = value
            self.left_child = None
            self.right_child = None

    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = BST.Node(value)
        else:
            self._insert(value, self.root)

    def _insert(self, value, current_node):
        if value < current_node.value:
            if current_node.left_child is None:
                current_node.left_child = BST.Node(value)
            else:
                self._insert(value, current_node.left_child)
        elif value > current_node.value:
            if current_node.right_child is None:
                current_node.right_child = BST.Node(value)
            else:
                self._insert(value, current_node.right_child)
        else:
            print("Value already exists in the tree.")

    def search(self, value):
        if self.root is None:
            return False
        else:
            return self._search(value, self.root)

    def _search(self, value, current_node):
        if current_node is None:
            return False
        elif current_node.value == value:
            return True
        elif value < current_node.value:
            return self._search(value, current_node.left_child)
        else:
            return self._search(value, current_node.right_child)

    def _inorder_traversal(self, node):
        if node:
            self._inorder_traversal(node.left_child)
            print(node.value)
            self._inorder_traversal(node.right_child)

    def print_tree(self):
        self._inorder_traversal(self.root)

    def sum_nodes(self):
        return self._sum_nodes_helper(self.root)

    def _sum_nodes_helper(self, node):
        if node is None:
            return 0
        return node.value + self._sum_nodes_helper(node.left_child) + self._sum_nodes_helper(node.right_child)

