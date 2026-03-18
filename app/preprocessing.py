import os
import subprocess
from datetime import datetime, timezone
from .models import db, Project
from .convert_sparse import convert_sparse_to_json
from .training import run_training


def run_preprocessing(app, project_id):

    with app.app_context():

        project = Project.query.get(project_id)

        project.status = "preprocessing"
        project.processing_started_at = datetime.now(timezone.utc)
        db.session.commit()

        base_path = os.path.join(app.config["UPLOAD_ROOT"], project_id)

        raw_path = os.path.join(base_path, "raw")
        processed_path = os.path.join(base_path, "processed")

        database_path = os.path.join(processed_path, "database.db")
        sparse_path = os.path.join(processed_path, "sparse")

        os.makedirs(sparse_path, exist_ok=True)

        try:

            subprocess.run([
                "colmap","feature_extractor",
                "--database_path", database_path,
                "--image_path", raw_path,
                "--ImageReader.single_camera","1"
            ], check=True)

            subprocess.run([
                "colmap","exhaustive_matcher",
                "--database_path", database_path
            ], check=True)

            subprocess.run([
                "colmap","mapper",
                "--database_path", database_path,
                "--image_path", raw_path,
                "--output_path", sparse_path
            ], check=True)

            print("COLMAP reconstruction finished")

            # FIND MODEL FOLDER
            models = os.listdir(sparse_path)

            if len(models) == 0:
                raise Exception("No sparse model created")

            model_path = os.path.join(sparse_path, models[0])

            print("Using sparse model:", model_path)

            points_bin = os.path.join(model_path, "points3D.bin")

            output_folder = os.path.join(base_path, "output")

            os.makedirs(output_folder, exist_ok=True)

            json_path = os.path.join(output_folder, "points.json")

            convert_sparse_to_json(points_bin, json_path)

            project.status = "ready_for_training"

            db.session.commit()

            # Auto-start training with default config
            run_training(app, project_id)
            return

        except Exception as e:

            print("ERROR:", e)

            project.status = "failed"

        db.session.commit()