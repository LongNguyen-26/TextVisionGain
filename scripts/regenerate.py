"""Regenerate the FJCAI paper's three tables and selection figure from aggregate CSVs.

Run from any directory with Python 3.12 and matplotlib 3.11.1 (the build versions).
With --compare-source DIR, also verify every CSV against the archived table directory.
Only tables/*.tex and figures/selection.pdf in this package are written.
"""
import argparse
import csv
from hashlib import sha256
from math import isclose
from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
CORPORA = ("jdocqa", "mpdocvqa")
NAMES = {"jdocqa": "JDocQA", "mpdocvqa": "MP-DocVQA"}
STRATA = ("1-2", "3-6", "7-20")
CATEGORIES = ("S1", "S0", "U")


def verify_data(original):
    for line in (ROOT / "data/SHA256SUMS").read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", maxsplit=1)
        directories = [ROOT / "data"] if original is None else [ROOT / "data", original]
        for directory in directories:
            file = directory / name
            if sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source digest mismatch: {file}")


def read(name):
    with (ROOT / "data" / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def unique(records, **keys):
    hits = [r for r in records if all(r[k] == v for k, v in keys.items())]
    if len(hits) != 1:
        raise ValueError(f"Expected exactly one row for {keys}, got {len(hits)}")
    return hits[0]


def nrange(lo, hi):
    return lo if lo == hi else f"{lo}--{hi}"


def point(value):
    return rf"\({float(value):.2f}\)"


def interval(row, low, high, scale):
    return rf"\([{scale * float(row[low]):.2f}, {scale * float(row[high]):.2f}]\)"


def write_table(name, header, rows, alignments):
    colspec = "@{}" + r"@{\enspace}".join(alignments) + "@{}"
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    for row in [header, None, *rows]:
        if row is None:
            lines.append(r"\midrule")
        else:
            lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ROOT / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


def engine_table():
    arms = read("earlier_ocr_arms")
    contrasts = read("earlier_ocr_contrasts")
    spec = [(arm, label) for arm, label in (
        ("T1", "Tesseract"), ("T2", "Tesseract + image"),
        ("P1", "PP-OCRv6"), ("P2", "PP-OCRv6 + image"), ("V", "Image only"))]
    means = {
        (corpus, arm): float(unique(arms, corpus=corpus, level="r4", arm=arm)["mean_anls_points"])
        for corpus in CORPORA for arm, _ in spec
    }
    rows = [[label, *[point(means[c, arm]) for c in CORPORA]] for arm, label in spec]
    rows.append(None)
    gain_t = [means[c, "T2"] - means[c, "T1"] for c in CORPORA]
    rows.append([r"Gain on Tesseract \(G_T\)", *map(point, gain_t)])
    for contrast, label in (("G_P", r"Gain on PP-OCRv6 \(G_P\)"), ("I", r"\(I=G_P-G_T\)")):
        values = [unique(contrasts, population="prespecified", corpus=c, contrast=contrast) for c in CORPORA]
        for c, g_t, value in zip(CORPORA, gain_t, values):
            g_p = means[c, "P2"] - means[c, "P1"]
            expected = g_p if contrast == "G_P" else g_p - g_t
            if not isclose(float(value["estimate_anls_points"]), expected, abs_tol=1e-10):
                raise ValueError(f"Engine contrast disagrees with arm means: {c} {contrast}")
        rows.append([label, *[point(v["estimate_anls_points"]) for v in values]])
        rows.append([r"98.75\% interval", *[interval(v, "marginal_98_75_low", "marginal_98_75_high", 1) for v in values]])
    write_table("engine", ["Input or contrast", *NAMES.values()], rows, "lrr")


def primary_table():
    records = read("main_primary")
    data = [unique(records, corpus=c, population="S1", labels="A") for c in CORPORA]
    rows = [["Qualified documents", *[v["n"] for v in data]]]
    arms = read("main_secondary")
    for arm, label in (("O1", "OCR alone"), ("O2", "OCR + images"),
                       ("R1", "Edited text alone"), ("R2", "Edited text + images"),
                       ("V", "Images alone")):
        values = [unique(arms, corpus=c, model="qwen", metric=arm) for c in CORPORA]
        rows.append([label, *[nrange(point(100 * float(v["std_lo"])), point(100 * float(v["std_hi"])))
                              for v in values]])
    rows.append(None)
    for prefix, label in (("G_O", r"Gain on OCR \(G_O\)"), ("G_R", r"Gain on edited text \(G_R\)"),
                          ("attenuation", r"\(D=G_O-G_R\)")):
        rows.append([label, *[nrange(point(100 * float(v[f"{prefix}_lo"])), point(100 * float(v[f"{prefix}_hi"])))
                               for v in data]])
    rows.append(None)
    rows.append(["Paired bootstrap", *[interval(v, "bootstrap_lo", "bootstrap_hi", 100) for v in data]])
    rows.append(["Simultaneous CP", *[interval(v, "simultaneous_CP_lo", "simultaneous_CP_hi", 100) for v in data]])
    if any(v["decision"] != "not_directionally_confirmed" for v in data):
        raise ValueError("Unexpected direction decision")
    write_table("primary", ["Input or contrast", *NAMES.values()], rows, "lrr")


def flip_table():
    records = read("main_opportunities")
    rows = []
    for corpus in CORPORA:
        for model, label in (("qwen", "Qwen"), ("internvl", "InternVL")):
            row = [NAMES[corpus], label]
            for metric in ("correction", "harm"):
                v = unique(records, corpus=corpus, model=model, baseline="R", metric=metric)
                event = nrange(v["numerator_lo"], v["numerator_hi"])
                opportunity = nrange(v["denominator_lo"], v["denominator_hi"])
                row.append(f"{event} of {opportunity}")
            rows.append(row)
    write_table("flips", ["Corpus", "Model", "Corrected", "Newly wrong"], rows, "llrr")


def selection_figure():
    flow = read("main_flow")
    sums = {}
    for corpus in CORPORA:
        for stratum in STRATA:
            group = [r for r in flow if (r["corpus"], r["pages"]) == (corpus, stratum)]
            count = tuple(sum(int(r[k]) for r in group) for k in CATEGORIES)
            if not group or sum(count) != sum(int(r["candidates"]) for r in group):
                raise ValueError(f"Invalid disposition counts: {corpus}, {stratum}")
            sums[corpus, stratum] = count
    if tuple(map(sum, zip(*sums.values()))) != (372, 37, 191):
        raise ValueError("Population does not match locked enrollment")

    with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 7.0, "pdf.fonttype": 42,
                         "hatch.linewidth": .4, "axes.linewidth": .45, "xtick.major.width": .45}):
        fig = Figure(figsize=(3.05, 2.24), facecolor="white")
        FigureCanvasAgg(fig)
        ax = fig.add_axes([.135, .195, .535, .775])
        counts = fig.add_axes([.706, .195, .287, .775])
        counts.set(xlim=(-.5, 2.5), ylim=(6.35, -1.05))
        counts.set_axis_off()
        ax.set(xlim=(0, 110), ylim=counts.get_ylim(), xticks=(0, 50, 110), yticks=[])
        ax.set_xlabel("Candidate documents", labelpad=2, fontsize=7)
        ax.tick_params(axis="x", length=2, pad=2, labelsize=6.8)
        ax.spines[["left", "right", "top"]].set_visible(False)
        colors = ("#656b50", "#b5b6af", "#f5f5f1")
        hatches = (None, None, "////")
        for i, (category, color, hatch) in enumerate(zip(CATEGORIES, colors, hatches)):
            counts.add_patch(Rectangle((i-.28, -.44), .56, .22, facecolor=color,
                                       edgecolor=".2", hatch=hatch, linewidth=.45))
            counts.text(i, -.77, category, ha="center", va="center", fontsize=7.0, weight="bold")
        for group_index, corpus in enumerate(CORPORA):
            start = group_index * 3.62
            ax.text(0, start-.31, NAMES[corpus], fontsize=7.5, weight="bold", va="center")
            for i, stratum in enumerate(STRATA):
                y = start + .35 + i
                ax.text(-4, y, stratum.replace("-", "\u2013"), ha="right", va="center")
                x = 0
                for col, (n, color, hatch) in enumerate(zip(sums[corpus, stratum], colors, hatches)):
                    ax.barh(y, n, left=x, height=.55, color=color, hatch=hatch,
                            edgecolor=".2", linewidth=.45)
                    counts.text(col, y, str(n), ha="center", va="center")
                    x += n
        fig.savefig(ROOT / "figures/selection.pdf", format="pdf",
                    metadata={"Title": "Reader qualification by corpus and length stratum",
                              "Creator": "FJCAI exhibit regeneration", "CreationDate": None, "ModDate": None})


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--compare-source", type=Path, metavar="DIR", help="also verify archived input CSVs")
    verify_data(args.parse_args().compare_source)
    (ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    engine_table()
    primary_table()
    flip_table()
    selection_figure()
