from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Project(db.Model):

    id = db.Column(db.String(36), primary_key=True)

    type = db.Column(db.String(10))

    status = db.Column(db.String(20))

    image_count = db.Column(db.Integer)

    valid_images = db.Column(db.Integer)

    invalid_images = db.Column(db.Integer)

    processing_started_at = db.Column(db.DateTime, nullable=True)

    training_started_at = db.Column(db.DateTime, nullable=True)

    training_iterations = db.Column(db.Integer, default=30000)

    training_max_resolution = db.Column(db.Integer, default=1024)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)