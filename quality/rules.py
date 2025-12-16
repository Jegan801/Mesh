def detect_mesh_errors(metrics, neighbors):
    errors = {}

    for elem_id, m in metrics.items():
        elem_errors = []

        # Rule 1: Very small area
        if m["area"] < 1.0:
            elem_errors.append("SMALL_AREA")

        # Rule 2: Bad aspect ratio
        if m["aspect_ratio"] > 3.0:
            elem_errors.append("BAD_ASPECT_RATIO")

        # Rule 3: Poor edge transition
        if m["edge_ratio"] > 3.0:
            elem_errors.append("BAD_TRANSITION")

        # Rule 4: Missing neighbors
        if len(neighbors.get(elem_id, [])) < 2:
            elem_errors.append("MISSING_NEIGHBOR")

        if elem_errors:
            errors[elem_id] = elem_errors

    return errors
