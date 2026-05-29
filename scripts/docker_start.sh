#!/bin/sh
set -eu

python -m src.pipeline
exec streamlit run app/streamlit_app.py
