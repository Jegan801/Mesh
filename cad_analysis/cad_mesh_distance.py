

import math


def _distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2 +
        (p1.z - p2.z) ** 2
    )


def compute_mesh_to_cad_distances(mesh, cad_mesh):
    """
    Computes minimum distance from each mesh node
    to nearest CAD node.

    Returns:
        dict: node_id -> min distance to CAD
    """

    cad_nodes = list(cad_mesh.nodes.values())
    distances = {}

    for nid, node in mesh.nodes.items():
        min_dist = float("inf")
        for cad_node in cad_nodes:
            d = _distance(node, cad_node)
            if d < min_dist:
                min_dist = d

        distances[nid] = min_dist

    print("Mesh-to-CAD node distances computed")

    return distances
