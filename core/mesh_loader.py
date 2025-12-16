

import csv


class Node:
    def __init__(self, node_id, x, y, z):
        self.id = int(float(node_id))
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def coords(self):
        return (self.x, self.y, self.z)


class Element:
    def __init__(self, elem_id, elem_type, node_ids):
        self.id = int(float(elem_id))
        self.type = elem_type  # TRI or QUAD
        self.node_ids = [int(float(nid)) for nid in node_ids]


class Mesh:
    def __init__(self):
        self.nodes = {}      # node_id -> Node
        self.elements = {}   # elem_id -> Element


def load_mesh(node_csv_path, element_csv_path):
    """
    Loads a mesh (CAD or first mesh) from CSV files.
    Handles:
    - TRI / QUAD elements
    - Float-formatted IDs (e.g. 19640.0)
    """

    mesh = Mesh()


    with open(node_csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if len(row) < 4:
                continue

            node_id, x, y, z = row[:4]
            try:
                node = Node(node_id, x, y, z)
                mesh.nodes[node.id] = node
            except ValueError:
                continue


    with open(element_csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if len(row) < 5:
                continue

            elem_id = row[0]
            elem_type = row[1].strip().upper()

            raw_node_ids = [nid for nid in row[2:] if nid.strip() != ""]

            if elem_type == "TRI":
                raw_node_ids = raw_node_ids[:3]
            elif elem_type == "QUAD":
                raw_node_ids = raw_node_ids[:4]
            else:
                continue

            try:
                elem = Element(elem_id, elem_type, raw_node_ids)
                mesh.elements[elem.id] = elem
            except ValueError:
                continue

    print(f"Loaded mesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} elements")

    return mesh
