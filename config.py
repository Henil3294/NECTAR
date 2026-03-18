import os

class Config:
    SECRET_KEY = "dev"
    SQLALCHEMY_DATABASE_URI = "sqlite:///database.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    UPLOAD_ROOT = os.path.join(os.getcwd(), "projects")

    IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
    VIDEO_EXTENSIONS = {"mp4", "mov", "avi"}

    NERF_ENV_PYTHON = os.path.join(os.getcwd(), "nerf_env", "Scripts", "python.exe")
    NS_TRAIN_PATH = os.path.join(os.getcwd(), "nerf_env", "Scripts", "ns-train.exe")
    DEFAULT_TRAINING_ITERATIONS = 15000
    DEFAULT_MAX_RESOLUTION = 1024