import matplotlib.pyplot as plt
import json
from pathlib import Path


to_plot = 'horizon_4_include_False'
with open(f"result/{to_plot}.json", "r") as f:
    data_full = json.load(f)
    data = data_full['result']
    
# fix according to to_run variable
name_d = {i:j for i, j in zip([2, 5, 10, 15], ["x2x2", "x5x5", "x10x10", "x15x15"])}

def name_change(name, include):
    for i in name_d:
        if include and name_d[i] in name:
            return i
        
        if not include and name_d[i] not in name:
            return i

include = False #fix accordingly
d = {}
for key in data:
    d[name_change(key, include)] = data[key]

new_d = {}
for i in d:
    dd = {}
    for j in d[i]:
        dd[name_change(j, True)] = d[i][j]
    new_d[i] = dd
print(new_d)

#Remove outliers manually here

# Define the data
import math
x = [2, 5, 10, 15]
bins = [[xs-0.3, xs-0.1, xs+0.1, xs+0.3] for xs in x]
values = [[math.log(sum(new_d[i][j])/len(new_d[i][j])) for j in new_d[i]] for i in new_d]

# Create the histogram
plt.figure(figsize=(8, 6))

for i, xs, bin in zip(range(4), x, bins):
    plt.bar([b[i] for b in bins], [val[i] for val in values], 0.2, label=str(xs))

# Customize the plot
plt.xlabel("Dataset")
plt.ylabel("Validation loss (log)")
plt.xticks(x)
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.savefig(f'eval_result/plots/{to_plot}.png')
plt.show()