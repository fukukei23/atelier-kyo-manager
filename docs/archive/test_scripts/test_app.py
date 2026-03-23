#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/yn441611/atelier-kyo-manager')

print("Testing Dashboard import...")
try:
    from app.web import create_app
    print("Dashboard import OK")
    app = create_app()
    print(f"App created: {app}")
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
