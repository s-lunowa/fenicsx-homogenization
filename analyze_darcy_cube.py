# Analyze the results from the Darcy flow with parametrized porosity
#
# Author: S.B. Lunowa

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epsilon", type=float, default=0.25, help="Averaging scaling epsilon.")
parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction, default=False, help="Enable verbose output.")
args = parser.parse_args()
if args.verbose:
    print(f"Options used: {args}", flush=True)

cases = ["isotropic", "anisotropic_p", "anisotropic_v", "anisotropic", "anisotropic_avg", "homogenized"]
names = [f"$\\mathrm{{K}}_{{{t}}}$" for t in ["\\rm iso", "\\rm p", "\\rm v", "\\phi", "\\rm\\overline{pv}", "\\rm hom}^{\\varepsilon"] ]

current_dir = os.path.dirname(os.path.abspath(__file__))

values = {
     "stokes": [0, 0, 0, 0]
}
with open(os.path.join(current_dir, "stokes", f"cube_cylinder", f"result.log")) as fin:
    while line := fin.readline():
        line = line.strip()
        if line.startswith("Total outflow"):
            values["stokes"][0] = float(line.split("[")[1].split()[0]) * 1e-3
        elif line.startswith("Average flow"):
            values["stokes"][1:] = line.split("[")[1].split(",")
            values["stokes"][3] = values["stokes"][3][:-1] # remove final ']'
        elif line.startswith("Volume"):
            volume = float(line.split()[1])
for i in range(3):
    values["stokes"][i+1] = float(values["stokes"][i+1]) * 1e-3 * volume

for i, case in enumerate(cases):
    values[case] = [0.0 for _ in range(4)]
    with open(os.path.join(current_dir, "darcy", f"cube_eps{args.epsilon}", f"{case}.log")) as fin:
        while line := fin.readline():
            line = line.strip()
            if line.startswith("Total in/outflow"):
                values[case][0] = float(line.split("[")[1].split()[0]) * 1e-3
            elif line.startswith("Average flow"):
                values[case][1:] = line.split("[")[1].split(",")
                values[case][3] = values[case][3][:-1] # remove final ']'
                for j in range(3):
                    values[case][j+1] = float(values[case][j+1]) * 1e-3
            elif line.startswith("L2 norm of divergence"):
                divergence = float(line.split()[-1])
                if divergence > 1e-12:
                    print(f"Warning: High divergence {divergence} for case {case}!")
    values["rel_err_" + case] = [(values[case][k] - values["stokes"][k]) / values["stokes"][k] for k in range(4)]

if args.verbose:
    for k,v in values.items():
        print(f"{k}: {v}")

scale = int(np.log10(values["stokes"][1])) -1
cases.insert(0, "stokes")
print("\\begin{tabular}{c|cc@{\\ }cc@{\\ }cc@{\\ }cc@{\\ }cc@{\\ }c}")
print("    \\toprule")
line = "    Quantity & Stokes "
for n in names:
    line += "& \\multicolumn{2}{c}{" + n + "} "
print(line + "\\\\")
print(r"    \midrule")
for i in range(4):
    line = "    "
    if i == 0:
        line += r"$v_{\rm out}$ "
    else:
        line += r"$\bar{v}_{" + str(i) + r"}$\ \ \ "
    line += r"[$\times 10^{" + str(scale) + r"}$]"
    for c in cases:
        line += " & " + f"${values[c][i] / 10**scale:.3f}$"
        if c != "stokes":
            rel_err = values["rel_err_" + c][i] * 100
            line += " & " + f"(${rel_err:+.0f}\\%$)"
    line += r" \\"
    print(line)
print(r"    \bottomrule")
print(r"\end{tabular}")

names.insert(0, "Stokes")
colors = ["black", "gray", "green!30!olive", "blue", "orange", "purple!50!violet", "brown!50!black"]
marks = ["*", "o", "diamond", "square", "triangle", "asterisk", "x"]

filename = os.path.join(current_dir, "darcy", f"cube_eps{args.epsilon}_average_velocity.tex")
with open(filename, "w", encoding="utf-8") as file:
    file.write("\\documentclass[tikz]{standalone}\n\\usepackage{pgfplots}\n\\usetikzlibrary{calc}\n\\begin{document}\n\\begin{tikzpicture}\n")
    for i in range(1,4):
        file.write("    \\begin{axis}[ xtick=data, legend pos=outer north east, grid=both, width=25mm, height=6cm,\n")
        file.write(f"                  {"title={Average velocity components $\\bar{v}_i$}, " if i == 2 else ""}")
        file.write(f"name=ax{i} {"" if i == 1 else ", at={($(ax" + str(i-1) + ".north east)+(7mm,0)$)}, anchor=north west"}]\n")
        for c, case in enumerate(cases):
            value_txt = f"({i}, {values[case][i]})"
            file.write(f"        \\addplot[{colors[c]}, mark={marks[c]}, only marks] coordinates {{ {value_txt} }};\n")
            if i == 3:
                file.write(f"        \\addlegendentry{{{names[c]}}}\n")
        file.write("    \\end{axis}\n")
    file.write("\\end{tikzpicture}\n\\end{document}")