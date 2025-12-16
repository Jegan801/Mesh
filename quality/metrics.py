import numpy as np


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def element_edges(node_coords):
    edges = []
    n = len(node_coords)
    for i in range(n):
        p1 = node_coords[i]
        p2 = node_coords[(i + 1) % n]
        edges.append(distance(p1, p2))
    return edges


def element_area(node_coords):
    # Triangle
    if len(node_coords) == 3:
        a, b, c = node_coords
        return 0.5 * np.linalg.norm(np.cross(
            np.array(b) - np.array(a),
            np.array(c) - np.array(a)
        ))

    # Quad -> split into two triangles
    elif len(node_coords) == 4:
        a, b, c, d = node_coords
        area1 = 0.5 * np.linalg.norm(np.cross(
            np.array(b) - np.array(a),
            np.array(c) - np.array(a)
        ))
        area2 = 0.5 * np.linalg.norm(np.cross(
            np.array(d) - np.array(a),
            np.array(c) - np.array(a)
        ))
        return area1 + area2

    return 0.0


def compute_quality_metrics(mesh):
    metrics = {}

    for elem_id, element in mesh.elements.items():
        coords = [mesh.nodes[nid].coords() for nid in element.node_ids]

        edges = element_edges(coords)
        area = element_area(coords)

        metrics[elem_id] = {
            "area": area,
            "aspect_ratio": max(edges) / min(edges) if min(edges) > 0 else 0,
            "edge_ratio": max(edges) / min(edges) if min(edges) > 0 else 0
        }

    return metrics
