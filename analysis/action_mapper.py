def map_actions(intrinsic_errors, cad_errors):
    actions = set()

    all_errors = set(intrinsic_errors) | set(cad_errors)

    if "ORPHAN_NODE" in all_errors:
        actions.add("DELETE")

    if "SMALL_AREA" in all_errors or "BAD_TRANSITION" in all_errors:
        actions.add("ADD")

    if (
        "BAD_ASPECT_RATIO" in all_errors
        or "HIGH_SKEWNESS" in all_errors
        or "CAD_DEVIATION_HIGH" in all_errors
    ):
        actions.add("MOVE")

    if "MISSING_NEIGHBOR" in all_errors or len(all_errors) >= 3:
        actions.add("REMESH")

    return list(actions)
