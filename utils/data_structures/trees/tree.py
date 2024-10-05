# tree.py

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

class Tree:
    def __init__(self, root):
        self.root = Node(root)

    def add_child(self, parent, child):
        if parent in self.get_values():
            node = self.find_node(parent)
            node.children.append(Node(child))

    def remove_child(self, parent, child):
        if parent in self.get_values():
            node = self.find_node(parent)
            node.children = [n for n in node.children if n.value != child]

    def get_values(self):
        values = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            values.append(node.value)
            stack.extend(node.children)
        return values

    def find_node(self, value):
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.value == value:
                return node
            stack.extend(node.children)
        return None

    def is_balanced(self):
        def height(node):
            if node is None:
                return 0
            return 1 + max(height(n) for n in node.children)

        return abs(height(self.root.children[0]) - height(self.root.children[1])) <= 1

    def is_bst(self):
        def is_bst_node(node, min_value, max_value):
            if node is None:
                return True
            if not min_value < node.value < max_value:
                return False
            return (is_bst_node(node.children[0], min_value, node.value) and
                    is_bst_node(node.children[1], node.value, max_value))

        return is_bst_node(self.root, float('-inf'), float('inf'))
