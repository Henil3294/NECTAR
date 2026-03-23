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

        # 1. Get the path to your engine
       # 1. Get the absolute path to your engine
      # 1. Get the absolute path to your engine
        compiler_script = os.path.abspath(os.path.join(os.getcwd(), '..', 'compile.py'))
        compiler_dir = os.path.dirname(compiler_script)
        
        
        raw_path_abs = os.path.abspath(raw_path)
        # We MUST make the image path absolute so the engine doesn't lose it when changing folders
       # 1. NEW: Create an absolute path for this project's specific output folder
        output_path_abs = os.path.abspath(output_path)

       # =================================================================
        # ☢️ THE FINAL STABLE VERSION: REAL TELEMETRY + VRAM OPTIMIZED
        # =================================================================
        bat_path = os.path.normpath(os.path.join(base_path, "run_engine.bat"))
        log_file_path = os.path.normpath(log_file_path)

        with open(bat_path, "w", encoding="utf-8") as f:
            f.write("@echo off\n")
            f.write("set PYTHONHOME=\n")
            f.write("set PYTHONPATH=\n")
            f.write(f'cd /d "{compiler_dir}"\n')
            f.write(r"call E:\NeRF_Studio\miniconda\Scripts\activate.bat gaussian" + "\n")
            
            # CHANGED: Using -r 4 for GTX 1650 stability. 
            # Redirecting all output (stdout and stderr) to training.log
            f.write(f'E:\\NeRF_Studio\\envs\\gaussian\\python.exe compile.py -i "{raw_path_abs}" -r 4 > "{log_file_path}" 2>&1\n')
            f.write("exit\n")

        print(f"\n[NECTAR] Launching Engine... Logs routing to: {log_file_path}\n")

        try:
            # 1. Clear old logs and start fresh
            with open(log_file_path, "w", encoding="utf-8") as init_log:
                init_log.write("[SYS] NECTAR Pipeline Initialized.\n")
                init_log.write("[SYS] Engine starting on GTX 1650...\n")

            # 2. Start the engine (Does NOT wait yet)
            process = subprocess.Popen(f'"{bat_path}"', shell=True)
            
            # 3. THE MONITOR: Update the progress JSON while the engine runs
            import time
            while process.poll() is None:  # While the batch file terminal is still open
                if os.path.exists(log_file_path):
                    with open(log_file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        if lines:
                            # Parse the very last line of the log for progress
                            metrics = parse_training_line(lines[-1])
                            if metrics:
                                # This updates the JSON file that triggers the blue wave!
                                update_progress_file(progress_path, metrics, iterations)
                
                time.sleep(1) # Check the log once every second

            # 4. Final check once the process actually finishes
            if process.returncode != 0:
                raise Exception(f"Engine exited with code {process.returncode}")

            print(f"Compilation success for {project_id}")
            project.status = "training_complete"

        except Exception as e:
            print(f"ERROR: {e}")
            project.status = "training_failed"
            
        
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