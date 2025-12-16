import plotly.graph_objects as go
from ai.hybrid_risk import hybrid_category


RISK_COLORS = {
    "LOW": "green",
    "MEDIUM": "orange",
    "HIGH": "red"
}


def plot_hybrid_risk(mesh, hybrid_risks, output_html="mesh_hybrid_risk.html"):
    fig = go.Figure()

    for elem_id, element in mesh.elements.items():
        coords = [mesh.nodes[nid].coords() for nid in element.node_ids]

        # close polygon
        xs = [c[0] for c in coords] + [coords[0][0]]
        ys = [c[1] for c in coords] + [coords[0][1]]

        score = hybrid_risks.get(elem_id, 0.0)
        category = hybrid_category(score)
        color = RISK_COLORS[category]

        hover_text = (
            f"<b>Element ID:</b> {elem_id}<br>"
            f"<b>Hybrid Risk Score:</b> {score:.2f}<br>"
            f"<b>Risk Level:</b> {category}"
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
            text=hover_text,
            showlegend=False
        ))

    fig.update_layout(
        title="Hybrid Mesh Quality Risk Visualization",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white"
    )

    fig.write_html(output_html)
    print(f"\nHybrid risk visualization saved to: {output_html}")
