from ai.risk_model import risk_category



def generate_scorecard(final_report):
    """
    final_report format:
    {
        element_id: {
            "severity": "HIGH" | "MEDIUM" | "LOW",
            "actions": ["DELETE", "ADD", ...],
            ...
        }
    }
    """

    severity_count = {
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0
    }

    action_count = {
        "DELETE": 0,
        "ADD": 0,
        "MOVE": 0,
        "REMESH": 0
    }

    for info in final_report.values():
        sev = info.get("severity", "LOW")
        severity_count[sev] += 1

        for action in info.get("actions", []):
            if action in action_count:
                action_count[action] += 1

    # Health score logic (simple & explainable)
    health_score = (
        100
        - severity_count["HIGH"] * 5
        - severity_count["MEDIUM"] * 2
        - severity_count["LOW"] * 1
    )

    health_score = max(0, min(100, health_score))

    return severity_count, action_count, health_score
