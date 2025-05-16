import pandas as pd
input_file = "all_calc.txt"
output_file = "data.xlsx"

data = []
current_method = None

with open(input_file, "r") as file:
    for line in file:
        line = line.strip()
        if line and not line[0].isdigit() and ":" not in line:
            current_method = line
            data.append([current_method, "", "", "", "", ""])
        elif ":" in line:
            parts = line.split()
            if len(parts) == 5:
                image_name = parts[0].replace(":", "")
                all_contours, tp, fp, fn = map(int, parts[1:])
                data.append(["", image_name, all_contours, tp, fp, fn])

df = pd.DataFrame(data, columns=["Method", "Image", "All Contours", "TP (True Positive)", "FP (False Positive)",
                                 "FN (False Negative)"])

df.to_excel(output_file, index=False)
print(f"Údaje boli uložené do {output_file}")
