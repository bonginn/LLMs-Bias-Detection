import matplotlib.pyplot as plt
from bias_score_compute import compute_bias_score

model_names = ["Llama-3-8B-Instruct"]
bias_scores = {}
for model in model_names:
    for i in range(1, 4):
        bias_score = compute_bias_score(f"assignments/{model}/Naive/results_{i}.json")["Bias Score"]
        print(f"{model} - {i} : {bias_score}")
        bias_scores[f"{model} - {i}"] = bias_score

# draw
models = list(bias_scores.keys())
scores = list(bias_scores.values())

plt.figure(figsize=(8, 5))
bars = plt.bar(models, scores, color=['#4C72B0', '#55A868', '#C44E52'])

# show values on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

plt.ylim(0, 1)
plt.title("Bias Score Comparison Across LLMs")
plt.ylabel("Bias Score")
plt.xlabel("Model")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
