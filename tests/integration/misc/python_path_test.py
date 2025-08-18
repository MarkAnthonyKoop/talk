import sys
import os

print("Environment PYTHONPATH:")
print(os.environ.get("PYTHONPATH", "(not set)"))
print("\nsys.path (Python import search paths):")
for path in sys.path:
    print(f"  {path}")

