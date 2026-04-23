import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from vector_store.faiss_store import load_faiss_artifacts
from agents.orchestrator import Orchestrator

# 1. Load index and metadata
index, metadata, manifest = load_faiss_artifacts("data/processed")

# 2. Create orchestrator
orchestrator = Orchestrator(index=index, metadata=metadata)

# 3. Run on a real product ID
result = orchestrator.run("M2M-FIRE")

# 4. Print result
print(json.dumps(result, indent=2))
