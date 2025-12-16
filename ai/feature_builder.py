
def build_feature_matrix(metrics, neighbors, errors):
    X = {}
    
    for elem_id, m in metrics.items():
        neighbor_count = len(neighbors.get(elem_id, []))
        error_count = len(errors.get(elem_id, [])) if elem_id in errors else 0

        X[elem_id] = [
            m["area"],
            m["aspect_ratio"],
            m["edge_ratio"],
            neighbor_count,
            error_count
        ]

    return X
