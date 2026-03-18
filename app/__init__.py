from flask import Flask
from .models import db
from .routes import upload_bp
from config import Config

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)

    db.init_app(app)

    with app.app_context():
        db.create_all()

    app.register_blueprint(upload_bp)

    return app