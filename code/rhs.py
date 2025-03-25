import json

with open("../results/deeponets/1/dataset1000.json", "r") as f:
    data = json.load(f)

a_k_list = [sample["a_k"] for sample in data]

with open("../results/deeponets/1/rhs_coeffs.json", "w") as f:
    json.dump(a_k_list, f)

print("Saved a_k values to rhs_coeffs.json")
