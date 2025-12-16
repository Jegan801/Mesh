def build_feature_vector(eid, mesh, metrics, intrinsic_errors, cad_errors):
    errors_i = intrinsic_errors.get(eid, [])
    errors_c = cad_errors.get(eid, [])

    m = metrics.get(eid, {})

    return [
        m.get("area", 0.0),
        m.get("aspect_ratio", 0.0),
        m.get("skewness_proxy", 0.0),
        m.get("min_edge", 0.0),
        m.get("max_edge", 0.0),
        len(mesh.element_neighbors.get(eid, [])),
        int("BAD_TRANSITION" in errors_i),
        int("MISSING_NEIGHBOR" in errors_i),
        int("SMALL_AREA" in errors_i),
        int("CAD_DEVIATION_HIGH" in errors_c),
        int("CAD_COVERAGE_WEAK" in errors_c),
    ]
