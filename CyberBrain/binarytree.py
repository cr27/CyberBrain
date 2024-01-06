class Node:
    def __init__(self, value):
        self.value = value
        self.left_child = None
        self.right_child = None

def add_node(root, value):
    if root is None:
        root = Node(value)
    else:
        if value < root.value:
            root.left_child = add_node(root.left_child, value)
        else:
            root.right_child = add_node(root.right_child, value)
    return root

def print_tree_preorder(root):
    if root is not None:
        print(root.value, end=' ')
        print_tree_preorder(root.left_child)
        print_tree_preorder(root.right_child)

#BST when we have weighted memories

