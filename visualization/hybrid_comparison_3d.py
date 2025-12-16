# ==========================================================
# SIDE-BY-SIDE 3D COMPARISON WITH NODES + ERROR INDEX
# ==========================================================

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ai.hybrid_risk import hybrid_category


RISK_COLORS = {
    "LOW": "green",
    "MEDIUM": "orange",
    "HIGH": "red"
}


def risk_summary(hybrid_risks):
    summary = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for score in hybrid_risks.values():
        summary[hybrid_category(score)] += 1
    return summary


def build_mesh_traces(mesh, hybrid_risks, show_nodes=True):
    x, y, z = [], [], []
    i, j, k = [], [], []
    face_colors = []

    node_index = {}
    idx = 0

    # Register nodes
    for nid, node in mesh.nodes.items():
        node_index[nid] = idx
        x.append(node.x)
        y.append(node.y)
        z.append(node.z)
        idx += 1

    # Build faces
    for elem_id, element in mesh.elements.items():
        ids = element.node_ids
        color = RISK_COLORS[hybrid_category(hybrid_risks.get(elem_id, 0.0))]

        if len(ids) == 3:
            i.append(node_index[ids[0]])
            j.append(node_index[ids[1]])
            k.append(node_index[ids[2]])
            face_colors.append(color)

        elif len(ids) == 4:
            i.append(node_index[ids[0]])
            j.append(node_index[ids[1]])
            k.append(node_index[ids[2]])
            face_colors.append(color)

            i.append(node_index[ids[0]])
            j.append(node_index[ids[2]])
            k.append(node_index[ids[3]])
            face_colors.append(color)

    surface = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        facecolor=face_colors,
        opacity=0.6,
        flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.8)
    )

    traces = [surface]

    # Wireframe
    ex, ey, ez = [], [], []
    for element in mesh.elements.values():
        ids = element.node_ids
        for t in range(len(ids)):
            p1 = mesh.nodes[ids[t]]
            p2 = mesh.nodes[ids[(t + 1) % len(ids)]]
            ex += [p1.x, p2.x, None]
            ey += [p1.y, p2.y, None]
            ez += [p1.z, p2.z, None]

    traces.append(go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        line=dict(color="black", width=1),
        name="Edges"
    ))

    # Nodes
    if show_nodes:
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=2, color="black"),
            name="Nodes"
        ))

    return traces


def plot_side_by_side(first_mesh, first_risks, final_mesh, final_risks, output_html):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("First Mesh (Hybrid Risk)", "Final Mesh (Hybrid Risk)")
    )

    # First mesh
    for trace in build_mesh_traces(first_mesh, first_risks):
        fig.add_trace(trace, row=1, col=1)

    # Final mesh
    for trace in build_mesh_traces(final_mesh, final_risks):
        fig.add_trace(trace, row=1, col=2)

    # Risk summaries
    first_summary = risk_summary(first_risks)
    final_summary = risk_summary(final_risks)

    annotation_text = (
        f"<b>First Mesh</b> — "
        f"🟢 {first_summary['LOW']} | "
        f"🟡 {first_summary['MEDIUM']} | "
        f"🔴 {first_summary['HIGH']}<br>"
        f"<b>Final Mesh</b> — "
        f"🟢 {final_summary['LOW']} | "
        f"🟡 {final_summary['MEDIUM']} | "
        f"🔴 {final_summary['HIGH']}"
    )

    fig.update_layout(
        title=dict(
            text="Hybrid Mesh Quality Comparison (First vs Final)",
            x=0.5
        ),
        annotations=[
            dict(
                text=annotation_text,
                x=0.5,
                y=1.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14)
            )
        ],
        scene=dict(aspectmode="data"),
        scene2=dict(aspectmode="data"),
        template="plotly_white"
    )

    fig.write_html(output_html)
    print(f"\nSide-by-side mesh comparison saved to: {output_html}")
