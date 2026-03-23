import os
import re
import json
from datetime import datetime, timezone


# Nerfstudio stdout format (with NO_COLOR / plain text):
# Step (% Done)       Train Iter (time)    Train Rays / Sec     Test PSNR            ETA (time)
# 100 (0.33%)         0.1234s              1.23M                22.1234              1h 2m 30s

# Updated to match: "Training progress: 2% | 720/30000"
# This stops exactly at the % sign, so it doesn't care if a pipe | or a space follows it.
STEP_PATTERN = re.compile(r"Training progress:\s*(\d+)%")

# Keep this for general time matching if needed
ETA_TIME_PATTERN = re.compile(
    r"(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?"
)


def parse_training_line(line):
    """Universal parser for the engine's output."""
    line = _strip_ansi(line).strip()
    if not line: return None

    # 1. Find the percentage first (e.g., "4%")
    pct_match = STEP_PATTERN.search(line)
    if not pct_match:
        return None

    percent = float(pct_match.group(1))

    # 2. Find the step numbers separately (e.g., "1270/30000")
    step_match = re.search(r"(\d+)/(\d+)", line)
    
    metrics = {
        "percent_done": percent,
        "current_step": int(step_match.group(1)) if step_match else 0,
        "total_steps": int(step_match.group(2)) if step_match else 0
    }

    # 3. Find the Loss if it exists
    loss_match = re.search(r"Loss=([0-9.]+)", line)
    if loss_match:
        metrics["loss"] = float(loss_match.group(1))

    return metrics


def update_progress_file(progress_path, metrics, total_steps):
    """Write training progress to a JSON file for the API to read."""
    metrics["total_steps"] = total_steps
    metrics["last_updated"] = datetime.now(timezone.utc).isoformat() + "Z"
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)


def read_progress(project_path):
    """Read the training progress JSON file. Returns dict or None."""
    progress_path = os.path.join(project_path, "logs", "training_progress.json")
    if not os.path.exists(progress_path):
        return None
    try:
        with open(progress_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _strip_ansi(text):
    """Remove all ANSI escape codes and control sequences from text."""
    # Remove all escape sequences: colors, cursor movement, erase, etc.
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    # Remove OSC sequences (e.g., hyperlinks)
    text = re.sub(r"\x1b\][^\x07]*\x07", "", text)
    # Remove any remaining escape characters
    text = re.sub(r"\x1b[^\x1b]*", "", text)
    # Remove carriage returns
    text = text.replace("\r", "")
    return text
