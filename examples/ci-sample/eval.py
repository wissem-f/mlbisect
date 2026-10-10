import json
import os

metric = 0.75 if os.environ.get("MLBISECT_CONFIG") else 1.0
print(json.dumps({"metric": metric}))
