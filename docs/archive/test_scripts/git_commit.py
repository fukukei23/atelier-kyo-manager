#!/usr/bin/env python3
import subprocess
import os

os.chdir("/home/yn441611/atelier-kyo-manager")

result = subprocess.run(
    ["git", "commit", "-m", "Refactor test_product_extractor.py with proper Mock setup"],
    capture_output=True,
    text=True
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("RETURNCODE:", result.returncode)
