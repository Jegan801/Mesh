

import math


def _distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2 +
        (p1.z - p2.z) ** 2
    )


def _triangle_area(n1, n2, n3):
    """
    Computes area of a triangle using cross product
    """
    ax = n2.x - n1.x
    ay = n2.y - n1.y
    az = n2.z - n1.z

    bx = n3.x - n1.x
    by = n3.y - n1.y
    bz = n3.z - n1.z

    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx

    return 0.5 * math.sqrt(cx**2 + cy**2 + cz**2)


def compute_intrinsic_metrics(mesh):
    """
    Computes intrinsic geometry metrics for each element.
    Returns:
        dict: elem_id -> metrics dict
    """

    metrics = {}

    for elem_id, elem in mesh.elements.items():
        nodes = [mesh.nodes[nid] for nid in elem.node_ids]

     
        edges = []
        for i in range(len(nodes)):
            n1 = nodes[i]
            n2 = nodes[(i + 1) % len(nodes)]
            edges.append(_distance(n1, n2))

        max_edge = max(edges)
        min_edge = min(edges)

        aspect_ratio = max_edge / min_edge if min_edge > 0 else float("inf")

        if len(nodes) == 3:
            area = _triangle_area(nodes[0], nodes[1], nodes[2])

        elif len(nodes) == 4:
            # Quad → split into two triangles
            area = (
                _triangle_area(nodes[0], nodes[1], nodes[2]) +
                _triangle_area(nodes[0], nodes[2], nodes[3])
            )
        else:
            area = 0.0


        skewness_proxy = aspect_ratio

        metrics[elem_id] = {
            "area": area,
            "edges": edges,
            "min_edge": min_edge,
            "max_edge": max_edge,
            "aspect_ratio": aspect_ratio,
            "skewness_proxy": skewness_proxy,
        }

    print("Intrinsic geometry metrics computed")

    return metrics
