from ai.risk_model import risk_category


def mesh_summary(metrics, errors, risks):
    total_elements = len(metrics)

    error_count = len(errors)

    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for score in risks.values():
        risk_counts[risk_category(score)] += 1

    avg_aspect = sum(m["aspect_ratio"] for m in metrics.values()) / total_elements

    return {
        "total_elements": total_elements,
        "error_elements": error_count,
        "risk_distribution": risk_counts,
        "avg_aspect_ratio": avg_aspect
    }


def compare_summaries(first, final):
    comparison = {}

    for key in first:
        if isinstance(first[key], dict):
            comparison[key] = {
                k: final[key][k] - first[key][k] for k in first[key]
            }
        else:
            comparison[key] = final[key] - first[key]

    return comparison
