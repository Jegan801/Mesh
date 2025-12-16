# =====================================================
# ERROR AGGREGATION & MESH IMPROVEMENT RECOMMENDATIONS
# =====================================================


def classify_severity(intrinsic_errors, cad_errors):
    """
    Classifies severity based on error combinations.
    """

    total_errors = len(intrinsic_errors) + len(cad_errors)

    if "CAD_DEVIATION_HIGH" in cad_errors:
        return "HIGH"

    if "SMALL_AREA" in intrinsic_errors and "BAD_ASPECT_RATIO" in intrinsic_errors:
        return "HIGH"

    if "MISSING_NEIGHBOR" in intrinsic_errors and cad_errors:
        return "HIGH"

    if total_errors >= 3:
        return "HIGH"

    if total_errors == 2:
        return "MEDIUM"

    if total_errors == 1:
        return "LOW"

    return "OK"


def generate_recommendations(intrinsic_errors, cad_errors):
    """
    Generates human-readable recommendations.
    """

    recs = []

    if "SMALL_AREA" in intrinsic_errors:
        recs.append("Increase local element size or remesh collapsed region.")

    if "BAD_ASPECT_RATIO" in intrinsic_errors:
        recs.append("Improve element shape by smoothing or remeshing.")

    if "HIGH_SKEWNESS" in intrinsic_errors:
        recs.append("Reduce skewness by adjusting node distribution.")

    if "BAD_TRANSITION" in intrinsic_errors:
        recs.append("Apply gradual mesh transition between adjacent regions.")

    if "MISSING_NEIGHBOR" in intrinsic_errors:
        recs.append("Check for holes or broken connectivity in the mesh.")

    if "CAD_DEVIATION_HIGH" in cad_errors:
        recs.append("Project mesh nodes closer to CAD surface.")

    if "CAD_COVERAGE_WEAK" in cad_errors:
        recs.append("Increase mesh density to better capture CAD geometry.")

    if not recs:
        recs.append("Mesh quality acceptable. No action required.")

    return recs


def aggregate_mesh_analysis(intrinsic_error_map, cad_error_map):
    """
    Combines intrinsic + CAD errors into final analysis.

    Returns:
        dict: elem_id -> analysis summary
    """

    final_report = {}

    all_elem_ids = set(intrinsic_error_map.keys()) | set(cad_error_map.keys())

    for eid in all_elem_ids:
        intrinsic = intrinsic_error_map.get(eid, [])
        cad = cad_error_map.get(eid, [])

        severity = classify_severity(intrinsic, cad)
        recommendations = generate_recommendations(intrinsic, cad)

        final_report[eid] = {
            "severity": severity,
            "intrinsic_errors": intrinsic,
            "cad_errors": cad,
            "recommendations": recommendations
        }

    print("Final mesh analysis & recommendations generated")

    return final_report
