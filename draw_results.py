import matplotlib.pyplot as plt
from bias_score_compute import compute_bias_score

model_names = [["Llama-3-8B-Instruct", "assign"], ["Llama-3-70B-Instruct", "assign"], ["Llama-4-Scout-17B-16E-Instruct", "assign"], ["GPT-4o-mini", "assign"]]
total_times = [5, 2, 1, 5]

bias_scores = {}
idx = 0
for model, mode in model_names:
    file = "Naive" if mode == "assign" else "reflected"
    total_bias_score = 0
    times = total_times[idx]
    idx += 1
    for i in range(1, times + 1):

        bias_score = compute_bias_score(f"assignments/{model}/{file}/results_{i}.json", mode)["Bias Score"]
        print(f"{model} - {i} : {bias_score}")
        total_bias_score += bias_score

    bias_scores[model] = total_bias_score / times  # Average bias score over 5 runs

# draw
models = list(bias_scores.keys())
scores = list(bias_scores.values())

plt.figure(figsize=(8, 5))
bars = plt.bar(models, scores, color=['#4C72B0', '#55A868', '#C44E52', "#DDF360"])

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
plt.savefig("chart/bias_score_comparison_total.png")  # Save the figure
plt.show()
