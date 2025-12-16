from ai.risk_model import risk_category


def compute_hybrid_risk(rule_risks, ml_probs):
    hybrid = {}

    for elem_id in rule_risks:
        rule_score = rule_risks.get(elem_id, 0.0)
        ml_score = ml_probs.get(elem_id, 0.0)

        final_score = 0.6 * rule_score + 0.4 * ml_score
        final_score = min(final_score, 1.0)

        hybrid[elem_id] = final_score

    return hybrid


def hybrid_category(score):
    if score >= 0.65:
        return "HIGH"
    elif score >= 0.35:
        return "MEDIUM"
    else:
        return "LOW"
