
MIN_AREA = 1.0                # Very small element area
MAX_ASPECT_RATIO = 3.0        # Distortion limit
MAX_SKEWNESS_PROXY = 3.0      # Same as aspect ratio proxy
MAX_AREA_TRANSITION = 3.0     
MIN_NEIGHBORS = 1             


def detect_intrinsic_errors(mesh, metrics, neighbors):
    """
    Detects intrinsic mesh errors using rule-based checks.

    Returns:
        dict: elem_id -> list of error strings
    """

    errors = {}

    for elem_id, m in metrics.items():
        elem_errors = []

        if m["area"] < MIN_AREA:
            elem_errors.append("SMALL_AREA")

        if m["aspect_ratio"] > MAX_ASPECT_RATIO:
            elem_errors.append("BAD_ASPECT_RATIO")

        if m["skewness_proxy"] > MAX_SKEWNESS_PROXY:
            elem_errors.append("HIGH_SKEWNESS")

        # -------------------------
        # Topology check
        # -------------------------
        if len(neighbors.get(elem_id, [])) <= MIN_NEIGHBORS:
            elem_errors.append("MISSING_NEIGHBOR")

        # -------------------------
        # Transition check
        # -------------------------
        nbrs = neighbors.get(elem_id, [])
        if nbrs:
            areas = [metrics[n]["area"] for n in nbrs if n in metrics]
            if areas:
                max_nbr_area = max(areas)
                min_nbr_area = min(areas)

                if min_nbr_area > 0:
                    transition_ratio = max_nbr_area / min_nbr_area
                    if transition_ratio > MAX_AREA_TRANSITION:
                        elem_errors.append("BAD_TRANSITION")

        errors[elem_id] = elem_errors

    print("Intrinsic mesh errors detected")

    return errors
