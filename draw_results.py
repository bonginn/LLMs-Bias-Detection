import matplotlib.pyplot as plt
from bias_score_compute import compute_bias_score

# model_names = [["Llama-3-8B-Instruct", "assign"], ["Llama-3-70B-Instruct", "assign"], ["Llama-4-Scout-17B-16E-Instruct", "assign"], ["GPT-4o-mini", "assign"]]
# total_times = [5, 3, 3, 5]

# model_names = [["GPT-4o-mini", "reflected"], ["Llama-3-8B-Instruct", "reflected"]]
# total_times = [3, 3]

# model_names = [["GPT-4o-mini", "gender_female"], ["GPT-4o-mini", "gender_male"], ["Llama-3-8B-Instruct", "gender_female"], ["Llama-3-8B-Instruct", "gender_male"]]
# total_times = [5, 5, 5, 5]


model_names = [["GPT-4o-mini", "assign"], ["GPT-4o-mini", "reflected"], ["GPT-4o-mini", "gender_female"], ["GPT-4o-mini", "g-reflected"]]
total_times = [5, 3, 5, 1]

bias_scores = {}
idx = 0
for model, mode in model_names:
    file = "Naive"
    if mode == "gender_female":
        file = "female"
    elif mode == "gender_male":
        file = "male"
    elif mode == "reflected":
        file = "Self-reflection"
    elif mode == "g-reflected":
        file = "Female-reflection"

    total_bias_score = 0
    times = total_times[idx]
    idx += 1
    for i in range(1, times + 1):

        bias_score = compute_bias_score(f"assignments/{model}/{file}/results_{i}.json", mode)["Bias Score"]
        print(f"{model} - {i} : {bias_score}")
        total_bias_score += bias_score

    bias_scores[model + " " + file] = total_bias_score / times  # Average bias score over 5 runs

# draw
models = list(bias_scores.keys())
scores = list(bias_scores.values())

plt.figure(figsize=(8, 5))
bars = plt.bar(models, scores, color=['#4C72B0', '#55A868', "#C7DB6C", "#D40C0C"])

# show values on bars
h = 0
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
    if yval < 0:
        h = -1


plt.ylim(h, 1)
plt.title(f"Differ Mode Bias Score Comparison Across GPT-4o-mini")
plt.ylabel("Bias Score")
plt.xlabel("Model")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("chart/total/bias_score_comparison_GPT-4o_gender-reflection.png")  # Save the figure
plt.show()
