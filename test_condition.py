import sys
sys.path.insert(0, "src")
from condition import check_condition
for f in ["torn.jpg", "clean.jpg"]:
    print(f, check_condition(f))