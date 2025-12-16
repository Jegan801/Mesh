

from collections import defaultdict


def _element_edges(node_ids):
    """
    Returns a set of edges for an element.
    Each edge is represented as a sorted tuple of node IDs.
    """
    edges = set()
    n = len(node_ids)

    for i in range(n):
        n1 = node_ids[i]
        n2 = node_ids[(i + 1) % n]
        edges.add(tuple(sorted((n1, n2))))

    return edges


def build_element_neighbors(mesh):
    """
    Builds an element-to-neighbor mapping.
    Two elements are neighbors if they share an edge.
    """

    edge_to_elements = defaultdict(list)

    for elem_id, elem in mesh.elements.items():
        edges = _element_edges(elem.node_ids)
        for edge in edges:
            edge_to_elements[edge].append(elem_id)


    neighbors = defaultdict(set)

    for edge, elems in edge_to_elements.items():
        if len(elems) > 1:
            for e in elems:
                neighbors[e].update(x for x in elems if x != e)

    for elem_id in mesh.elements:
        neighbors[elem_id] = neighbors.get(elem_id, set())

    print("Element neighbor graph built")

    return neighbors
