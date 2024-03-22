def save_summary(summary:dict, model_name:str):
    with open(f"./results/summary_{model_name}.txt", "w") as f:
        for key in summary.keys():
            f.write(f"{key}: {summary[key]}\n")
    return None