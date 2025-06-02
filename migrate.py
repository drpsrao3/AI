# migrate.py
from app import app, db
from flask_migrate import upgrade, migrate, init, stamp

with app.app_context():
    try:
        # Initialize migration repository if it doesn't exist
        init()
    except Exception as e:
        print(f"Migration repository already initialized: {e}")

    # Create a stamp for the initial migration
    stamp()

    # Generate migration scripts for the current model
    migrate(message="Initial migration")

    # Apply migrations to the database
    upgrade()