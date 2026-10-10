import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lab.datagen import evaluate_case

axis_values = {
    key.removeprefix("MLBISECT_AXIS_").lower(): value
    for key, value in os.environ.items()
    if key.startswith("MLBISECT_AXIS_")
}
metric = evaluate_case("L03_interaction", axis_values, int(os.environ["MLBISECT_SEED"]))
print(json.dumps({"metric": metric}))
