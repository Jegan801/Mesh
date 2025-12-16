# =====================================================
# CAD-BASED ERROR RULES
# Converts CAD distance metrics into mesh error labels
# =====================================================

# -----------------------------
# Thresholds (tunable)
# -----------------------------
MAX_NODE_CAD_DISTANCE = 2.5      # Allowed mesh-to-CAD deviation
MAX_ELEMENT_CAD_DISTANCE = 3.0   # Average element deviation


def detect_cad_related_errors(mesh, mesh_to_cad_distances):
    """
    Detects CAD-related mesh errors.

    Returns:
        dict: elem_id -> list of CAD error strings
    """

    cad_errors = {}

    for elem_id, elem in mesh.elements.items():
        node_distances = [
            mesh_to_cad_distances.get(nid, 0.0)
            for nid in elem.node_ids
        ]

        elem_errors = []

        if not node_distances:
            cad_errors[elem_id] = elem_errors
            continue

        max_node_dist = max(node_distances)
        avg_node_dist = sum(node_distances) / len(node_distances)

        # -------------------------
        # CAD deviation checks
        # -------------------------
        if max_node_dist > MAX_NODE_CAD_DISTANCE:
            elem_errors.append("CAD_DEVIATION_HIGH")

        if avg_node_dist > MAX_ELEMENT_CAD_DISTANCE:
            elem_errors.append("CAD_COVERAGE_WEAK")

        cad_errors[elem_id] = elem_errors

    print("CAD-related mesh errors detected")

    return cad_errors
