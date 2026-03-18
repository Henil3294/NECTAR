import os
import re
import json
from datetime import datetime, timezone


# Nerfstudio stdout format (with NO_COLOR / plain text):
# Step (% Done)       Train Iter (time)    Train Rays / Sec     Test PSNR            ETA (time)
# 100 (0.33%)         0.1234s              1.23M                22.1234              1h 2m 30s

STEP_PATTERN = re.compile(
    r"(\d+)\s+\((\d+\.\d+)%\)"
)

ETA_TIME_PATTERN = re.compile(
    r"(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?"
)


def parse_training_line(line):
    """Parse a nerfstudio stdout training line and extract metrics.

    Returns dict with available metrics, or None if line is not a training step.
    """
    line = _strip_ansi(line).strip()

    if not line:
        return None

    match = STEP_PATTERN.search(line)
    if not match:
        return None

    step = int(match.group(1))
    percent = float(match.group(2))

    # Skip header lines or non-step lines
    if step == 0 and percent == 0.0:
        return None

    metrics = {
        "current_step": step,
        "percent_done": percent,
    }

    # Extract all tokens after the step/percent match
    rest = line[match.end():]
    tokens = rest.split()

    for i, token in enumerate(tokens):
        # Extract time values (e.g., "0.1234s")
        if token.endswith("s") and not token.endswith("ps"):
            try:
                float(token[:-1])
                # This is train iter time, skip
                continue
            except ValueError:
                pass

        # Extract rays/sec (e.g., "1.57M" or "157K")
        if token.endswith("M"):
            try:
                metrics["rays_per_sec"] = float(token[:-1]) * 1e6
                continue
            except ValueError:
                pass
        elif token.endswith("K"):
            try:
                metrics["rays_per_sec"] = float(token[:-1]) * 1e3
                continue
            except ValueError:
                pass

        # Extract PSNR — a standalone float typically between 5 and 60
        try:
            val = float(token)
            if 5.0 <= val <= 60.0:
                metrics["psnr"] = val
                continue
        except ValueError:
            pass

    # Extract ETA from the end of the line (e.g., "1h 2m 30s", "45m 10s", "30s")
    eta_match = re.search(r"(?:(\d+)h)?\s*(?:(\d+)m)?\s*(\d+)s\s*$", line)
    if eta_match and eta_match.group(3):
        hours = int(eta_match.group(1)) if eta_match.group(1) else 0
        minutes = int(eta_match.group(2)) if eta_match.group(2) else 0
        seconds = int(eta_match.group(3))
        metrics["eta_seconds"] = hours * 3600 + minutes * 60 + seconds

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
