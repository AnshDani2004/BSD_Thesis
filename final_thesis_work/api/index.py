import sys
import os

# 1. Resolve project root
# When deploying from 'final_thesis_work', the parent of 'api' is the root.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. Add dashboard backend to path
backend_dir = os.path.join(project_root, 'dashboard', 'backend')
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# 3. Import the Flask instance
from app import app

# Export for Vercel
app = app
