# =====================================================
# CAE-STYLE 3D MESH ERROR DEBUG VISUALIZATION
# =====================================================

import plotly.graph_objects as go
from collections import Counter

ERROR_COLORS = {
    "BAD_ASPECT_RATIO": "red",
    "BAD_TRANSITION": "orange",
    "SMALL_AREA": "purple",
    "MISSING_NEIGHBOR": "black",
    "OK": "green"
}


def element_centroid(mesh, element):
    xs, ys, zs = [], [], []
    for nid in element.node_ids:
        n = mesh.nodes[nid]
        xs.append(n.x)
        ys.append(n.y)
        zs.append(n.z)
    return sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)


def plot_mesh_errors_3d(mesh, errors, output_html="mesh_error_debug.html"):
    fig = go.Figure()
    error_counter = Counter()

    # ---------- Draw faces ----------
    for elem_id, element in mesh.elements.items():
        coords = [mesh.nodes[nid] for nid in element.node_ids]
        xs = [n.x for n in coords] + [coords[0].x]
        ys = [n.y for n in coords] + [coords[0].y]
        zs = [n.z for n in coords] + [coords[0].z]

        if elem_id in errors:
            err_type = errors[elem_id][0]
            color = ERROR_COLORS[err_type]
            error_counter[err_type] += 1
        else:
            color = ERROR_COLORS["OK"]

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=color, width=4),
            opacity=0.9,
            showlegend=False
        ))

    # ---------- Missing neighbor markers ----------
    for elem_id, err_list in errors.items():
        if "MISSING_NEIGHBOR" in err_list:
            cx, cy, cz = element_centroid(mesh, mesh.elements[elem_id])
            fig.add_trace(go.Scatter3d(
                x=[cx], y=[cy], z=[cz],
                mode="markers",
                marker=dict(size=10, color="black", symbol="x"),
                showlegend=False
            ))

    # ---------- Nodes ----------
    nx, ny, nz = [], [], []
    for node in mesh.nodes.values():
        nx.append(node.x)
        ny.append(node.y)
        nz.append(node.z)

    fig.add_trace(go.Scatter3d(
        x=nx, y=ny, z=nz,
        mode="markers",
        marker=dict(size=6, color="blue", opacity=0.8),
        name="Nodes"
    ))

    # ---------- Error index ----------
    index_text = "<b>Mesh Error Index</b><br>"
    for err, cnt in error_counter.items():
        index_text += f"<span style='color:{ERROR_COLORS[err]}'>{err}: {cnt}</span><br>"

    fig.update_layout(
        title="CAE-Style Mesh Error Visualization",
        annotations=[dict(
            text=index_text,
            x=0.01, y=0.99,
            xref="paper", yref="paper",
            showarrow=False,
            align="left",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )],
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        template="plotly_white"
    )

    fig.write_html(output_html)
    print(f"Mesh error debug visualization saved to {output_html}")
