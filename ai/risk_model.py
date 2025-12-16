# ai/risk_model.py

def compute_risk_scores(features):
    risks = {}

    for elem_id, f in features.items():
        area, aspect, edge, neighbors, errors = f

        risk = 0.0

        # Aspect ratio contribution
        if aspect > 3:
            risk += 0.4
        elif aspect > 2:
            risk += 0.2

        # Edge transition contribution
        if edge > 3:
            risk += 0.3
        elif edge > 2:
            risk += 0.15

        # Connectivity contribution
        if neighbors < 2:
            risk += 0.2

        # Error count contribution
        risk += min(errors * 0.1, 0.3)

        # Clamp to [0, 1]
        risk = min(risk, 1.0)

        risks[elem_id] = risk

    return risks


def risk_category(score):
    if score >= 0.6:
        return "HIGH"
    elif score >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"
