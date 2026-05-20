"""Tests conftest.py - ensures project root is in sys.path for test imports."""
import sys
import os

# Add project root to sys.path so that modules like 'evaluation', 'model', 'training'
# can be imported in tests without requiring package installation.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
