# Analyze the results from the Darcy flow with averaged porosity
#
# Author: S.B. Lunowa

import argparse
import numpy as np
import os

cases = ["isotropic", "anisotropic_p", "anisotropic_v", "anisotropic", "anisotropic_avg"]
names = [f"$\\mathrm{{K}}_{{{t}}}$" for t in ["\\rm iso", "\\rm p", "\\rm v", "\\phi", "\\rm\\overline{pv}"] ]
colors = ["orange!80!yellow", "green!30!olive", "violet!50!purple", "gray", "blue"]
styles = ["solid", "densely dashed"]
markers = ["o", "diamond", "square", "triangle", "star"]
current_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--summary", action=argparse.BooleanOptionalAction, default=False, help="Whether to summarize the results in one plot.")
parser.add_argument("-c", "--case", choices=cases, default=cases[0])
args = parser.parse_args()
print(f"Options used: {args}", flush=True)

radii = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
boundaries = ["avg", "fluid", "reflection", "wall"]
if args.summary:
    boundaries = ["wall"]
filters = ["box", "gaussian"]

figstart = """\\begin{tikzpicture}
    \\begin{semilogyaxis}[
        title={TITLE},
        xmin=0.2, xmax=0.55, xtick={0.25, 0.3, 0.35, 0.4, 0.45, 0.5}, xlabel={REV radius $r_{\\rm REV}$},
        domain=0.24:0.51, samples=2, legend pos=outer north east, legend cell align=left,
        yminorticks, grid=both
    ]
        \\addplot[very thick] { VALUE };
        \\addlegendentry{Stokes}
"""

figs = ["", "", "", ""]
stokes = [ 0.0, 0.0, 0.0, 0.0 ]

with open(os.path.join(current_dir, "stokes", "cylinder_spiral", "result.log")) as fin:
    while line := fin.readline():
        line = line.strip()
        if line.startswith("Total outflow"):
            stokes[0] = float(line.split("[")[1].split()[0]) * 1e-3
        elif line.startswith("Average flow"):
            stokes[1:] = line.split("[")[1].split(",")
            stokes[3] = stokes[3][:-1] # remove final ']'
        elif line.startswith("Volume"):
            volume = float(line.split()[1])
for i in range(3):
    stokes[i+1] = float(stokes[i+1]) * 1e-3 * volume / (3 * np.pi)

for i, fig in enumerate(figs):
    replacement = "Total outflow" if i == 0 else f"Average velocity $\\bar{{v}}_{i}$"
    if not args.summary:
        replacement += ", " + names[cases.index(args.case)]
    figs[i] = figstart.replace("TITLE", replacement).replace("VALUE", str(stokes[i]))
    if i > 1:
        figs[i] = figs[i].replace("semilogy", "")

if not args.summary:
    cases = [args.case]

for c, case in enumerate(cases):
    for bnd in boundaries:
        color = "gray" if bnd == "fluid" else "green!30!olive" if bnd == "wall" else "blue" if bnd == "avg" else "orange"
        for filter in filters:
            mark = "square*" if filter == "box" else "*, mark options={solid}"
            style = styles[0] if filter == "box" else styles[1]
            if args.summary:
                color = colors[c]
                mark = markers[c]
                if filter == "box":
                    if c == 0:
                        mark = "*"
                    elif c == 4:
                        mark = "asterisk"
                    else:
                        mark += "*"
            for i in range(len(figs)):
                figs[i] += "        \\addplot[" + color + ", thick, " + style + ", mark=" + mark + ", mark options={solid}] coordinates {\n"

            for radius in radii:
                values = [0.0 for _ in range(4)]
                with open(os.path.join(current_dir, "darcy", f"cylinder_{case}", f"REV_{radius}_Filter_{filter}_BC_{bnd}.log")) as fin:
                    while line := fin.readline():
                        line = line.strip()
                        if line.startswith("Total outflow"):
                            values[0] = float(line.split("[")[1].split()[0]) * 1e-3
                        elif line.startswith("Average flow"):
                            values[1:] = line.split("[")[1].split(",")
                            values[3] = values[3][:-1] # remove final ']'
                        elif line.startswith("L2 norm of divergence"):
                            divergence = float(line.split()[-1])
                            if divergence > 1e-9:
                                print(f"Warning: High divergence {divergence} for r = {radius}, bnd = {bnd}, filter = {filter}")
                    for i in range(3):
                        values[i+1] = float(values[i+1]) * 1e-3
                for i in range(len(figs)):
                    figs[i] += f"            ({radius}, {values[i]})\n"
            for i in range(len(figs)):
                if args.summary:
                    figs[i] += f"        }};\n        \\addlegendentry{{{names[c]}, {filter}}}\n"
                else:
                    figs[i] += f"        }};\n        \\addlegendentry{{$\\rho_{{\\rm {bnd}}}$, {filter}}}\n"

figend = "    \\end{semilogyaxis}\n\\end{tikzpicture}\n"
if args.summary:
    filename = os.path.join(current_dir, "darcy", "cylinder_spiral_average_velocity.tex")
else:
    filename = os.path.join(current_dir, "darcy", f"cylinder_spiral_{args.case}_average_velocity.tex")
with open(filename, "w", encoding="utf-8") as file:
    file.write("\\documentclass[tikz]{standalone}\n\\usepackage{pgfplots}\n\\begin{document}\n")
    for i, fig in enumerate(figs):
        file.write(fig)
        file.write(figend if i <= 1 else figend.replace("semilogy", ""))
    file.write("\n\\end{document}")
