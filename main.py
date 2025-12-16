from core.mesh_loader import load_mesh
from core.mesh_neighbors import build_element_neighbors
from quality.intrinsic_metrics import compute_intrinsic_metrics
from quality.intrinsic_rules import detect_intrinsic_errors
from cad_analysis.cad_mesh_distance import compute_mesh_to_cad_distances
from cad_analysis.cad_rules import detect_cad_related_errors

from ml.feature_builder import build_feature_vector
from ml.severity_predictor import predict_severity

from analysis.action_mapper import map_actions
from analysis.scorecard import generate_scorecard

from visualization.mesh_visualizer import visualize_first_mesh_edges
from ui.dashboard_template import render_dashboard



def main():
    print("[INFO] Loading First Mesh...")
    mesh = load_mesh(
        "data_2/01_/first_mesh_2_NODE.csv",
        
        "data_2/01_/first_mesh_2_ELEMENT.csv"
    )

    print("[INFO] Loading CAD (analysis only)...")
    cad = load_mesh(
        "data_2/01_/cad_NODE.csv",
        "data_2/01_/cad_ELEMENT.csv"
    )

    print("[INFO] Computing intrinsic mesh metrics...")
    mesh.element_neighbors = build_element_neighbors(mesh)
    metrics = compute_intrinsic_metrics(mesh)
    intrinsic_errors = detect_intrinsic_errors(
        mesh, metrics, mesh.element_neighbors
    )

    print("[INFO] Computing CAD vs mesh deviation...")
    distances = compute_mesh_to_cad_distances(mesh, cad)
    cad_errors = detect_cad_related_errors(mesh, distances)

    print("[INFO] Running AI severity prediction...")
    final_report = {}

    severity_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

    for eid in mesh.elements:
        features = build_feature_vector(
            eid,
            mesh,
            metrics,
            intrinsic_errors,
            cad_errors
        )

        ml_class, confidence = predict_severity(features)
        severity = severity_map[ml_class]

        actions = map_actions(
            intrinsic_errors.get(eid, []),
            cad_errors.get(eid, [])
        )

        final_report[eid] = {
            "severity": severity,
            "confidence": confidence,
            "intrinsic_errors": intrinsic_errors.get(eid, []),
            "cad_errors": cad_errors.get(eid, []),
            "actions": actions
        }

        print(
            f"Element {eid} → "
            f"Severity: {severity}, "
            f"Actions: {actions}, "
            f"Confidence: {confidence}"
        )

    print("[INFO] Generating mesh visualization...")
    visualize_first_mesh_edges(
        mesh,
        final_report,
        out_html="html/first_mesh_ml_severity.html"
    )


    print("[INFO] Generating recommendations dashboard...")
    severity_count, action_count, health_score = generate_scorecard(final_report)

    render_dashboard(
        severity_count,
        action_count,
        health_score,
        output_path="html/recommendations_dashboard.html"
    )

    print("[DONE] AI Mesh Quality Copilot completed successfully!")


if __name__ == "__main__":
    main()
