#!/bin/bash
flask db upgrade
gunicorn --bind 0.0.0.0:10000 --workers 2 --threads 4 --timeout 120 app:app