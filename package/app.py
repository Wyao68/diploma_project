"""Streamlit app entrypoint for Streamlit Community Cloud.

This file intentionally keeps minimal logic: importing `predict.py` will execute
its Streamlit UI code. Streamlit will use this file as the app root.
"""

# Ensure project root is importable
import os
import sys
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importing predict runs the Streamlit UI defined there
from predict import *

# usage example - streamlit run "C:\Users\86153\Desktop\diploma_project\package\app.py"
