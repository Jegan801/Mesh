import sys
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

from Unsupervised.ml.feature_builder import build_feature_vector
from Unsupervised.core.mesh_loader import load_mesh
from Unsupervised.core.mesh_neighbors import build_element_neighbors
from Unsupervised.quality.intrinsic_metrics import compute_intrinsic_metrics
from Unsupervised.quality.intrinsic_rules import detect_intrinsic_errors
from Unsupervised.cad_analysis.cad_mesh_distance import compute_mesh_to_cad_distances
from Unsupervised.cad_analysis.cad_rules import detect_cad_related_errors


def evaluate(vehicle_id: str, mesh_idx: str):
    vehicle_root = Path("data_2") / vehicle_id
    model_path = Path("models") / vehicle_id / "severity_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[INFO] Evaluating vehicle {vehicle_id}, mesh {mesh_idx}")

    # ---- Load model ----
    model = joblib.load(model_path)

    # ---- Load CAD ----
    cad = load_mesh(
        vehicle_root / "cad_NODE.csv",
        vehicle_root / "cad_ELEMENT.csv"
    )

    # ---- Load mesh ----
    mesh = load_mesh(
        vehicle_root / f"first_mesh_{mesh_idx}_NODE.csv",
        vehicle_root / f"first_mesh_{mesh_idx}_ELEMENT.csv"
    )

    # ---- Feature computation ----
    mesh.element_neighbors = build_element_neighbors(mesh)
    metrics = compute_intrinsic_metrics(mesh)
    intrinsic_errors = detect_intrinsic_errors(mesh, metrics, mesh.element_neighbors)
    distances = compute_mesh_to_cad_distances(mesh, cad)
    cad_errors = detect_cad_related_errors(mesh, distances)

    X = []
    y_true = []

    for eid in mesh.elements:
        feat = build_feature_vector(
            eid, mesh, metrics, intrinsic_errors, cad_errors
        )
        X.append(feat)

        # ---- Rule-based teacher ----
        if (
            "BAD_ASPECT_RATIO" in intrinsic_errors.get(eid, [])
            or "HIGH_SKEWNESS" in intrinsic_errors.get(eid, [])
            or "CAD_DEVIATION_HIGH" in cad_errors.get(eid, [])
        ):
            y_true.append(2)
        elif "BAD_TRANSITION" in intrinsic_errors.get(eid, []):
            y_true.append(1)
        else:
            y_true.append(0)

    X = np.asarray(X, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int64)

    # ---- Predict ----
    y_pred = model.predict(X)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m Unsupervised.ml.evaluate_model <vehicle_id> <mesh_idx>")
        print("Example: python -m Unsupervised.ml.evaluate_model 01_ 2")
        sys.exit(1)

    evaluate(sys.argv[1], sys.argv[2])
