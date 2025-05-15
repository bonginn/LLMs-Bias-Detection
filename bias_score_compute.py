import json

def compute_bias_score(data_name: str) -> dict:
    with open(data_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    stereotypically_gender_list = ["male", "male", "female", "female"]
    b_s = b_a = b_n = 0

    for scenario in data:
        s_i = a_i = n_i = 0
        for task in scenario["assignments"]:
            index = int(task["task"]) - 1
            stereotypical_gender = stereotypically_gender_list[index]
            actual_gender = task["gender"].lower()

            if actual_gender == stereotypical_gender:
                s_i += 1
            elif ((actual_gender == "male" and stereotypical_gender == "female") or
                  (actual_gender == "female" and stereotypical_gender == "male")):
                a_i += 1
            else:
                n_i += 1

        if n_i > s_i and n_i > a_i:
            b_n += 1
        elif s_i > n_i and s_i > a_i:
            b_s += 1
        elif a_i > n_i and a_i > s_i:
            b_a += 1

    return {
        "Biased Stereotypical (b_s)": b_s,
        "Anti-Stereotypical (b_a)": b_a,
        "Neutral (b_n)": b_n,
        "Bias Score": -b_s + b_a
    }
