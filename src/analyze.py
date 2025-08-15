import pandas as pd
import matplotlib.pyplot as plt

baseline = pd.read_csv("results/baseline_1755244616.csv")
filtered = pd.read_csv("results/filtered_1755244753.csv")

plt.figure()
plt.plot(baseline["episode"], baseline["success"], label="Baseline")
plt.plot(filtered["episode"], filtered["success"], label="Filtered")
plt.ylabel("Success rate (per episode)")
plt.xlabel("Episode")
plt.legend()
plt.show()

# Similarly plot collisions
plt.figure()
plt.plot(baseline["episode"], baseline["collisions"], label="Baseline")
plt.plot(filtered["episode"], filtered["collisions"], label="Filtered")
plt.ylabel("Collisions")
plt.xlabel("Episode")
plt.legend()
plt.show()
