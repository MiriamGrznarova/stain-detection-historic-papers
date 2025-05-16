import math


def calculate_width_height_ratio(vykal_width, vykal_height, listina_width, listina_height):
    maxi_sam = max(vykal_width, vykal_height)
    maxi_lis = max(listina_width, listina_height)
    ratio = (maxi_sam) / (maxi_lis)
    return ratio

# Definovanie dát
data = [
    (10, 10, 2422, 1665),  # 10x10 vykal, listina 2422x1665
    (7, 7, 2220, 1665),    # 7x7 vykal, listina 2220x1665
    (10, 16, 1756, 2803),  # 10x16 vykal, listina 1756x2803
    (7, 6, 1756, 2803),    # 7x6 vykal, listina 1756x2803
    (6, 4, 1756, 2803),    # 6x4 vykal, listina 1756x2803
    (13, 18, 1588, 3336),  # 13x18 vykal, listina 1588x3336
    (15, 29, 1588, 3336),  # 15x29 vykal, listina 1588x3336
    (14, 14, 2449, 1651),  # 14x14 vykal, listina 2449x1651
    (10, 9, 2449, 1651)    # 10x9 vykal, listina 2449x1651
]

ratios = [calculate_width_height_ratio(vw, vh, lw, lh) for vw, vh, lw, lh in data]

min_ratio = min([r for r in ratios])
max_ratio = max([r for r in ratios])


print("Pomer pre každý prípad (šírka, výška):")
for i, ratio in enumerate(ratios, 1):
    print(f"Prípad {i}: {ratio:.70f}")

print(f"\nMinimálny pomer: {min_ratio:.100f}")
print(f"Maximálny pomer: {max_ratio:.100f}")

print("\nRozmery vykalu:", math.sqrt(max_ratio * 1280*600))