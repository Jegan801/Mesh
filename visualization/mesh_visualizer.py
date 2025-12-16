import os
import plotly.graph_objects as go

SEVERITY_COLOR = {
    "HIGH": "red",
    "MEDIUM": "orange",
    "LOW": "yellow"
}


def visualize_first_mesh_edges(mesh, final_report, out_html):
    fig = go.Figure()

    bx, by, bz = [], [], []

    for elem in mesh.elements.values():
        nodes = elem.node_ids
        for i in range(len(nodes)):
            n1 = mesh.nodes[nodes[i]]
            n2 = mesh.nodes[nodes[(i + 1) % len(nodes)]]
            bx += [n1.x, n2.x, None]
            by += [n1.y, n2.y, None]
            bz += [n1.z, n2.z, None]

    fig.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz,
        mode="lines",
        line=dict(color="black", width=1),
        name="First Mesh Edges"
    ))

    # -------------------------------------------------
    # 2. Mesh nodes (debug visibility)
    # -------------------------------------------------
    nx = [n.x for n in mesh.nodes.values()]
    ny = [n.y for n in mesh.nodes.values()]
    nz = [n.z for n in mesh.nodes.values()]

    fig.add_trace(go.Scatter3d(
        x=nx, y=ny, z=nz,
        mode="markers",
        marker=dict(size=2, color="black"),
        name="Mesh Nodes"
    ))

    # -------------------------------------------------
    # 3. Problematic elements (edges highlighted)
    # -------------------------------------------------
    for severity in ["HIGH", "MEDIUM", "LOW"]:
        ex, ey, ez = [], [], []

        for eid, info in final_report.items():
            if info["severity"] != severity:
                continue

            elem = mesh.elements[eid]
            nodes = elem.node_ids

            for i in range(len(nodes)):
                n1 = mesh.nodes[nodes[i]]
                n2 = mesh.nodes[nodes[(i + 1) % len(nodes)]]
                ex += [n1.x, n2.x, None]
                ey += [n1.y, n2.y, None]
                ez += [n1.z, n2.z, None]

        if ex:
            fig.add_trace(go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode="lines",
                line=dict(color=SEVERITY_COLOR[severity], width=4),
                name=f"{severity} Severity"
            ))

    # -------------------------------------------------
    # Layout
    # -------------------------------------------------
    fig.update_layout(
        title="First Mesh Quality Analysis (CAD vs First Mesh)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        legend=dict(x=0.02, y=0.98)
    )

    # -------------------------------------------------
    # Save HTML
    # -------------------------------------------------
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html)
    print(f"[OK] Visualization saved to {out_html}")
