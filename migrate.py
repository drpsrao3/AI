# migrate.py
from app import app, db
from flask_migrate import upgrade, migrate, init
import os

with app.app_context():
    try:
        if not os.path.exists('migrations'):
            print("Initializing migration repository...")
            init()
        else:
            print("Migration repository already initialized")
        print("Generating migration scripts...")
        migrate(message="Create user table")
        print("Applying migrations...")
        upgrade()
        print("Migrations applied successfully")
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        raise