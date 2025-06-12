#!/bin/bash
export FLASK_APP=app.py
echo "Applying migrations..."
python migrate.py || { echo "Migration failed"; exit 1; }
echo "Starting gunicorn..."
gunicorn -w 2 -b 0.0.0.0:$PORT app:app