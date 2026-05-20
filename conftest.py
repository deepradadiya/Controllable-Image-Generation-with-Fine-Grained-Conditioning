"""Root conftest.py - ensures project root is in sys.path for test imports."""
import sys
import os

# Add project root to sys.path so that modules like 'evaluation', 'model', 'training'
# can be imported in tests without requiring package installation.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
