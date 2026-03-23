import os
import uuid
from flask import Blueprint, json, request, jsonify, current_app,render_template , send_file
from werkzeug.utils import secure_filename
from .models import db, Project
from .utils import create_project_structure, validate_image
from .preprocessing import run_preprocessing
from .training import run_training
from .training_monitor import read_progress
from threading import Thread
import psutil
import GPUtil
from flask import jsonify

upload_bp = Blueprint("upload", __name__)


def allowed_file(filename, allowed_set):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in allowed_set


@upload_bp.route("/")
def home():
    return render_template("upload.html")


@upload_bp.route("/upload", methods=[ "GET", "POST"])
def upload():

    if "images" not in request.files:
        return jsonify({"error": "No images provided"}), 400

    files = request.files.getlist("images")

    project_id = str(uuid.uuid4())

    project_path = os.path.join(
        current_app.config["UPLOAD_ROOT"],
        project_id
    )

    create_project_structure(project_path)

    raw_path = os.path.join(project_path, "raw")

    valid_images = 0
    invalid_images = 0

    for file in files:

        if not allowed_file(
            file.filename,
            current_app.config["IMAGE_EXTENSIONS"]
        ):
            invalid_images += 1
            continue

        filename = secure_filename(file.filename)

        save_path = os.path.join(raw_path, filename)

        file.save(save_path)

        is_valid = True

        if is_valid:
            valid_images += 1
        else:
            invalid_images += 1
            os.remove(save_path)

    if valid_images < 20:
        status = "invalid"
    else:
        status = "valid_images"

    project = Project(
        id=project_id,
        type="image",
        status=status,
        image_count=len(files),
        valid_images=valid_images,
        invalid_images=invalid_images
    )

    db.session.add(project)
    db.session.commit()

    if status == "valid_images":

        app = current_app._get_current_object()

        Thread(target=run_preprocessing, args=(app, project_id)).start()

        

    return jsonify({
        "project_id": project_id,
        "status": status,
        "valid_images": valid_images,
        "invalid_images": invalid_images
    })

@upload_bp.route("/status/<project_id>")
def status_page(project_id):

    project = Project.query.get(project_id)

    if not project:
        return "Project not found", 404

    return render_template("status.html", project=project)


@upload_bp.route("/api/status/<project_id>")
def get_status(project_id):

    project = Project.query.get(project_id)

    if not project:
        return {"error": "Project not found"}, 404

    started = None
    if project.processing_started_at:
        started = project.processing_started_at.isoformat() + "Z"

    training_started = None
    if project.training_started_at:
        training_started = project.training_started_at.isoformat() + "Z"

    return {
        "project_id": project.id,
        "status": project.status,
        "valid_images": project.valid_images,
        "processing_started_at": started,
        "training_started_at": training_started,
        "training_iterations": project.training_iterations,
        "training_max_resolution": project.training_max_resolution
    }


@upload_bp.route("/pointcloud/<project_id>")
def pointcloud(project_id):

    base_path = os.path.join(
        current_app.config["UPLOAD_ROOT"],
        project_id
    )

    ply_path = os.path.join(base_path, "output", "points.ply")

    if not os.path.exists(ply_path):
        return {"error": "Point cloud not ready"}, 404

    return send_file(ply_path)


@upload_bp.route("/points/<project_id>")
def get_points(project_id):
    # 🟢 FORCE STATIC PATH: We are pointing directly to the workspace for now
    static_splat_path = r"E:\Neural_Scene_Compiler\workspace\processing\model_output\point_cloud\iteration_30000\point_cloud.ply"

    if os.path.exists(static_splat_path):
        # We ignore the project_id and just serve the tree from the workspace
        return send_file(static_splat_path, mimetype='application/octet-stream')
    else:
        # If this happens, your workspace is empty or path is wrong
        print(f"CRITICAL: Static splat file not found at {static_splat_path}")
        return {"error": "Static splat file not found in workspace!"}, 404
    
@upload_bp.route("/viewer/<project_id>")
def viewer(project_id):

    return render_template(
        "viewer.html",
        project_id=project_id
    )


@upload_bp.route("/api/train/<project_id>", methods=["POST"])
def start_training(project_id):

    project = Project.query.get(project_id)

    if not project:
        return {"error": "Project not found"}, 404

    allowed = {"ready_for_training", "training_failed", "training_complete"}
    if project.status not in allowed:
        return {"error": f"Cannot train: status is {project.status}"}, 400

    data = request.get_json(silent=True) or {}
    iterations = data.get("iterations", current_app.config["DEFAULT_TRAINING_ITERATIONS"])
    max_resolution = data.get("max_resolution", current_app.config["DEFAULT_MAX_RESOLUTION"])

    project.training_iterations = iterations
    project.training_max_resolution = max_resolution
    db.session.commit()

    app = current_app._get_current_object()
    Thread(target=run_training, args=(app, project_id)).start()

    return {
        "project_id": project.id,
        "status": "training",
        "training_iterations": iterations,
        "training_max_resolution": max_resolution
    }


@upload_bp.route("/api/training-progress/<project_id>")
def training_progress(project_id):

    project = Project.query.get(project_id)

    if not project:
        return {"error": "Project not found"}, 404

    base_path = os.path.join(
        current_app.config["UPLOAD_ROOT"],
        project_id
    )

    progress = read_progress(base_path)

    return {
        "project_id": project.id,
        "status": project.status,
        "progress": progress
    }


@upload_bp.route("/api/logs/<project_id>")
def get_logs(project_id):

    project = Project.query.get(project_id)
    if not project:
        return {"error": "Project not found"}, 404

    base_path = os.path.join(
        current_app.config["UPLOAD_ROOT"],
        project_id
    )

    log_path = os.path.join(base_path, "logs", "training.log")
    if not os.path.exists(log_path):
        return {"lines": [], "status": project.status}

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except IOError:
        return {"lines": [], "status": project.status}

    # Return the last 50 lines
    tail = [l.rstrip("\n\r") for l in all_lines[-50:]]

    return {"lines": tail, "status": project.status}

# ... (your existing routes are here) ...

@upload_bp.route('/api/hardware', methods=['GET'])
def hardware_stats():
    # 1. CPU & System RAM
    cpu_load = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024**3)
    ram_total_gb = ram.total / (1024**3)

    # 2. NVIDIA GPU & VRAM
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_load = gpu.load * 100
        vram_used_mb = gpu.memoryUsed
        vram_total_mb = gpu.memoryTotal
        vram_percent = (vram_used_mb / vram_total_mb) * 100
    else:
        # Fallback if GPU isn't detected for a moment
        gpu_load, vram_used_mb, vram_total_mb, vram_percent = 0, 0, 0, 0

    return jsonify({
        'cpu_load': round(cpu_load, 1),
        'ram_used': round(ram_used_gb, 1),
        'ram_total': round(ram_total_gb, 1),
        'ram_percent': ram.percent,
        'gpu_load': round(gpu_load, 1),
        'vram_used': round(vram_used_mb / 1024, 2), # Convert MB to GB
        'vram_total': round(vram_total_mb / 1024, 2),
        'vram_percent': round(vram_percent, 1)
    })