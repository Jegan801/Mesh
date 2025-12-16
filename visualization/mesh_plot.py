# visualization/mesh_plot.py

import plotly.graph_objects as go


ERROR_COLORS = {
    "BAD_ASPECT_RATIO": "red",
    "BAD_TRANSITION": "orange",
    "SMALL_AREA": "purple",
    "MISSING_NEIGHBOR": "black"
}


def plot_mesh(mesh, errors, output_html="mesh_quality.html"):
    fig = go.Figure()

    for elem_id, element in mesh.elements.items():
        coords = [mesh.nodes[nid].coords() for nid in element.node_ids]
        xs = [c[0] for c in coords] + [coords[0][0]]
        ys = [c[1] for c in coords] + [coords[0][1]]

        if elem_id in errors:
            err_type = errors[elem_id][0]
            color = ERROR_COLORS.get(err_type, "red")
            hover = f"Element {elem_id}<br>Errors: {errors[elem_id]}"
        else:
            color = "lightgray"
            hover = f"Element {elem_id}<br>Status: OK"

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            fill="toself",
            mode="lines",
            line=dict(color=color),
            fillcolor=color,
            opacity=0.7,
            hoverinfo="text",
            text=hover,
            showlegend=False
        ))

    fig.update_layout(
        title="Mesh Quality Visualization",
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    fig.write_html(output_html)
    print(f"Mesh visualization saved to {output_html}")
