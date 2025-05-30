import matplotlib.pyplot as plt
from bias_score_compute import compute_bias_score

category = {    
    "Team dyn":list(range(0, 17)),
    "Famliy":list(range(17, 32)),
    "Office":list(range(32, 48)),
    "Hospital":list(range(48, 61)),
    "Politics":list(range(61, 77)),
    "Legal":list(range(77, 93)),
    "School":list(range(93, 111))
}

model = "GPT-4o-mini"  # Change this to the desired model
mode = "assign"
file = "Naive" if mode == "assign" else "reflected"

category_bias_scores = {}

for cat, indices in category.items():
    total_bias_score = 0
    for i in range(1, 6):
        bias_score = compute_bias_score(f"assignments/{model}/{file}/results_{i}.json", mode, indices[0], indices[-1])["Bias Score"]
        print(f"{cat} - {i} : {bias_score}")
        total_bias_score += bias_score

    category_bias_scores[cat] = total_bias_score / 5  # Average bias score over the category

labels = list(category_bias_scores.keys())
scores = [category_bias_scores[cat] for cat in labels]

fig, ax = plt.subplots()
bars = ax.bar(labels, scores)
ax.set_ylabel("Average Bias Score")
ax.set_title("Bias Scores by Category")
plt.xticks(rotation=45, ha='right')

# Annotate bars with numeric values
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2, 
        height, 
        f"{height:.2f}", 
        ha='center', 
        va='bottom'
    )

plt.tight_layout()
plt.savefig(f"chart/{model}/{model}_cate.png")  # Save the figure
plt.show()
