# visualization/risk_plot.py

import plotly.graph_objects as go
from ai.risk_model import risk_category

RISK_COLORS = {
    "LOW": "green",
    "MEDIUM": "orange",
    "HIGH": "red"
}


def plot_risk_zones(mesh, risks, output_html="mesh_risk_zones.html"):
    fig = go.Figure()

    for elem_id, element in mesh.elements.items():
        coords = [mesh.nodes[nid].coords() for nid in element.node_ids]
        xs = [c[0] for c in coords] + [coords[0][0]]
        ys = [c[1] for c in coords] + [coords[0][1]]

        score = risks.get(elem_id, 0.0)
        category = risk_category(score)
        color = RISK_COLORS[category]

        hover = (
            f"Element ID: {elem_id}<br>"
            f"Risk Score: {score:.2f}<br>"
            f"Risk Level: {category}"
        )

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            fill="toself",
            mode="lines",
            line=dict(color=color),
            fillcolor=color,
            opacity=0.75,
            hoverinfo="text",
            text=hover,
            showlegend=False
        ))

    fig.update_layout(
        title="Mesh Quality Risk Zones",
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    fig.write_html(output_html)
    print(f"Risk zone visualization saved to {output_html}")

