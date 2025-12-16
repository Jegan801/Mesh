# ==========================================
# CAE-STYLE 3D HYBRID RISK VISUALIZATION
# (Faces + Wireframe + Proper Lighting)
# ==========================================

import plotly.graph_objects as go
from ai.hybrid_risk import hybrid_category

RISK_COLORS = {
    "LOW": "green",
    "MEDIUM": "orange",
    "HIGH": "red"
}


def plot_hybrid_risk_3d(mesh, hybrid_risks, output_html):
    x, y, z = [], [], []
    i, j, k = [], [], []
    face_colors = []

    node_index = {}
    idx = 0

    # ----------------------------
    # 1. Register nodes
    # ----------------------------
    for nid, node in mesh.nodes.items():
        node_index[nid] = idx
        x.append(node.x)
        y.append(node.y)
        z.append(node.z)
        idx += 1

    # ----------------------------
    # 2. Build triangular faces
    # ----------------------------
    for elem_id, element in mesh.elements.items():
        ids = element.node_ids
        color = RISK_COLORS[hybrid_category(hybrid_risks.get(elem_id, 0.0))]

        # Triangle
        if len(ids) == 3:
            i.append(node_index[ids[0]])
            j.append(node_index[ids[1]])
            k.append(node_index[ids[2]])
            face_colors.append(color)

        # Quad → split into 2 triangles
        elif len(ids) == 4:
            i.append(node_index[ids[0]])
            j.append(node_index[ids[1]])
            k.append(node_index[ids[2]])
            face_colors.append(color)

            i.append(node_index[ids[0]])
            j.append(node_index[ids[2]])
            k.append(node_index[ids[3]])
            face_colors.append(color)

    # ----------------------------
    # 3. Surface (faces)
    # ----------------------------
    mesh_surface = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        facecolor=face_colors,
        opacity=0.65,   # IMPORTANT: see through
        lighting=dict(
            ambient=0.5,
            diffuse=0.8,
            roughness=0.4,
            specular=0.3
        ),
        flatshading=True
    )

    # ----------------------------
    # 4. Wireframe (edges)
    # ----------------------------
    edge_x, edge_y, edge_z = [], [], []

    for element in mesh.elements.values():
        ids = element.node_ids
        n = len(ids)

        for t in range(n):
            n1 = ids[t]
            n2 = ids[(t + 1) % n]

            p1 = mesh.nodes[n1]
            p2 = mesh.nodes[n2]

            edge_x += [p1.x, p2.x, None]
            edge_y += [p1.y, p2.y, None]
            edge_z += [p1.z, p2.z, None]

    mesh_edges = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="black", width=1),
        name="Mesh Edges"
    )

    # ----------------------------
    # 5. Final figure
    # ----------------------------
    fig = go.Figure(data=[mesh_surface, mesh_edges])

    fig.update_layout(
        title="CAE-Style 3D Hybrid Mesh Risk Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        template="plotly_white"
    )

    fig.write_html(output_html)
    print(f"\nCAE-style 3D hybrid risk visualization saved to: {output_html}")
