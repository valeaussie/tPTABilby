import os
import json
from collections import Counter

# Directory containing the JSON files
base_dir = "/fred/oz005/users/vdimarco/tBilby/300_sims_tight_uniforms_2/p_values"

# Initialise the counter
rank_counter = Counter()

# Loop over all model files
for i in range(299):  # adjust range if needed
    filename = os.path.join(base_dir, f"p_values_model_{i}.json")
    
    if not os.path.exists(filename):
        continue  # skip missing files
    
    with open(filename, "r") as f:
        data = json.load(f)
    
    injected = str(data["injected"])  # injected model index
    del data["injected"]  # remove to rank only model keys

    # Sort models by p-value, highest first
    sorted_models = sorted(data.items(), key=lambda x: x[1], reverse=True)

    # Find rank of injected
    for rank, (model, _) in enumerate(sorted_models, start=1):
        if model == injected:
            rank_counter[rank] += 1
            break

# Print results
for rank in sorted(rank_counter):
    print(f"Injected model ranked #{rank}: {rank_counter[rank]} times")
