import os
import re
import sys
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# ABSOLUTE IMPORTS (your structure)
# -----------------------------
from Unsupervised.core.mesh_loader import load_mesh
from Unsupervised.core.mesh_neighbors import build_element_neighbors
from Unsupervised.quality.intrinsic_metrics import compute_intrinsic_metrics
from Unsupervised.quality.intrinsic_rules import detect_intrinsic_errors
from Unsupervised.cad_analysis.cad_mesh_distance import compute_mesh_to_cad_distances
from Unsupervised.cad_analysis.cad_rules import detect_cad_related_errors
from Unsupervised.ml.feature_builder import build_feature_vector


# -----------------------------
# DISCOVER FIRST MESH PAIRS
# -----------------------------
def find_first_meshes(vehicle_root: Path):
    """
    Returns list of (NODE.csv, ELEMENT.csv) pairs
    """
    pairs = []

    for node_csv in vehicle_root.glob("first_mesh_*_NODE.csv"):
        m = re.search(r"first_mesh_(\d+)_NODE.csv", node_csv.name)
        if not m:
            continue

        idx = m.group(1)
        elem_csv = vehicle_root / f"first_mesh_{idx}_ELEMENT.csv"

        if elem_csv.exists():
            pairs.append((node_csv, elem_csv))

    return sorted(pairs)



def main(vehicle_id: str):
    print(f"[INFO] Training vehicle: {vehicle_id}")

    vehicle_root = Path("data_2") / vehicle_id
    if not vehicle_root.exists():
        raise FileNotFoundError(f"Vehicle folder not found: {vehicle_root}")

    # ---- Load CAD ONCE ----
    cad = load_mesh(
        vehicle_root / "cad_NODE.csv",
        vehicle_root / "cad_ELEMENT.csv"
    )
    print("[INFO] CAD loaded")

    mesh_pairs = find_first_meshes(vehicle_root)
    print(f"[INFO] Found {len(mesh_pairs)} first meshes")

    if not mesh_pairs:
        raise RuntimeError("No first meshes found for training")

    X_all = []
    y_all = []

    # ---- Iterate all first meshes ----
    for node_csv, elem_csv in mesh_pairs:
        print(f"[INFO] Processing {node_csv.name}")

        mesh = load_mesh(node_csv, elem_csv)
        mesh.element_neighbors = build_element_neighbors(mesh)

        metrics = compute_intrinsic_metrics(mesh)
        intrinsic_errors = detect_intrinsic_errors(
            mesh, metrics, mesh.element_neighbors
        )

        distances = compute_mesh_to_cad_distances(mesh, cad)
        cad_errors = detect_cad_related_errors(mesh, distances)

        for eid in mesh.elements:
            features = build_feature_vector(
                eid, mesh, metrics, intrinsic_errors, cad_errors
            )

            X_all.append(features)

            # ---- RULE-BASED LABELS (current truth source) ----
            if (
                "BAD_ASPECT_RATIO" in intrinsic_errors.get(eid, [])
                or "HIGH_SKEWNESS" in intrinsic_errors.get(eid, [])
                or "CAD_DEVIATION_HIGH" in cad_errors.get(eid, [])
            ):
                y_all.append(2)   # HIGH
            elif "BAD_TRANSITION" in intrinsic_errors.get(eid, []):
                y_all.append(1)   # MEDIUM
            else:
                y_all.append(0)   # LOW

    # ---- Convert to NumPy ----
    X = np.asarray(X_all, dtype=np.float32)
    y = np.asarray(y_all, dtype=np.int64)

    print(f"[INFO] Training samples: {X.shape}")

    # ---- Train Model ----
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        n_jobs=1,             
        random_state=42
    )
    model.fit(X, y)

    # ---- Save Model ----
    out_dir = Path("models") / vehicle_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "severity_model.pkl"
    joblib.dump(model, model_path)

    print(f"[OK] Model saved to {model_path}")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m Unsupervised.ml.train_severity_model <vehicle_id>")
        print("Example: python -m Unsupervised.ml.train_severity_model 01_")
        sys.exit(1)

    vehicle_id = sys.argv[1]
    main(vehicle_id)
