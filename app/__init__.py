from flask import Flask
from .models import db
from .routes import upload_bp
from config import Config

def create_app():
    # 🟢 Pointing to the correct folders
    app = Flask(__name__, 
                instance_relative_config=True,
                static_folder='../static',
                template_folder='templates')

    # 🟢 Loading the database configuration
    app.config.from_object(Config)

    # Initialize database
    db.init_app(app)

    with app.app_context():
        db.create_all()

    # Register the blueprint for routes
    app.register_blueprint(upload_bp)

    return app