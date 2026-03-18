import os
import subprocess
import cv2
import math
from datetime import datetime, timezone
from .models import db, Project
from .training_monitor import parse_training_line, update_progress_file


def create_downscaled_images(raw_path, factor):
    """Pre-create downscaled images in raw_{factor}/ folder for nerfstudio."""
    if factor <= 1:
        return

    dest_dir = raw_path.rstrip("/\\") + f"_{factor}"
    if os.path.isdir(dest_dir) and len(os.listdir(dest_dir)) > 0:
        return  # Already exists

    os.makedirs(dest_dir, exist_ok=True)

    for fname in os.listdir(raw_path):
        src = os.path.join(raw_path, fname)
        if not os.path.isfile(src):
            continue

        img = cv2.imread(src)
        if img is None:
            continue

        h, w = img.shape[:2]
        new_w = math.floor(w / factor)
        new_h = math.floor(h / factor)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(dest_dir, fname), resized)


def run_training(app, project_id):

    with app.app_context():

        project = Project.query.get(project_id)

        project.status = "training"
        project.training_started_at = datetime.now(timezone.utc)
        db.session.commit()

        base_path = os.path.join(app.config["UPLOAD_ROOT"], project_id)
        logs_path = os.path.join(base_path, "logs")
        output_path = os.path.join(base_path, "output")
        progress_path = os.path.join(logs_path, "training_progress.json")
        log_file_path = os.path.join(logs_path, "training.log")

        os.makedirs(logs_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        iterations = project.training_iterations or app.config["DEFAULT_TRAINING_ITERATIONS"]
        max_res = project.training_max_resolution or app.config["DEFAULT_MAX_RESOLUTION"]

        # Compute downscale factor from user-selected max resolution
        # nerfstudio downscale_factor must be a power of 2 (1, 2, 4)
        downscale_map = {2048: 1, 1024: 2, 512: 4}
        downscale_factor = downscale_map.get(max_res, 2)

        # Pre-create downscaled images so nerfstudio doesn't prompt interactively
        raw_path = os.path.join(base_path, "raw")
        if downscale_factor > 1:
            create_downscaled_images(raw_path, downscale_factor)

        ns_train = app.config["NS_TRAIN_PATH"]

        cmd = [
            ns_train, "nerfacto",
            "--data", base_path,
            "--output-dir", output_path,
            "--max-num-iterations", str(iterations),
            "--vis", "tensorboard",
            "--logging.local-writer.max-log-size", "0",
            "colmap",
            "--images-path", "raw",
            "--colmap-path", os.path.join("processed", "sparse", "0"),
            "--downscale-factor", str(downscale_factor),
        ]

        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"
            env["NO_COLOR"] = "1"
            env["TERM"] = "dumb"

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                env=env,
                encoding="utf-8",
                errors="replace",
            )

            with open(log_file_path, "w", encoding="utf-8") as log_file:
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()

                    metrics = parse_training_line(line)
                    if metrics:
                        update_progress_file(progress_path, metrics, iterations)

            process.wait()

            if process.returncode != 0:
                raise Exception(f"ns-train exited with code {process.returncode}")

            print(f"Training complete for project {project_id}")

            project.status = "training_complete"

        except Exception as e:

            print(f"Training ERROR for {project_id}: {e}")

            error_path = os.path.join(logs_path, "training_error.log")
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(str(e))

            project.status = "training_failed"

        db.session.commit()
