class Node:
    def __init__(self, node_id, x, y, z):
        self.id = int(node_id)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def coords(self):
        return (self.x, self.y, self.z)


class Element:
    def __init__(self, elem_id, node_ids):
        self.id = int(elem_id)
        self.node_ids = [int(nid) for nid in node_ids]


class Mesh:
    def __init__(self):
        self.nodes = {}      # node_id -> Node
        self.elements = {}   # elem_id -> Element
