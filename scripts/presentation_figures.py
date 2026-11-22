"""Draw the study design, ANLS interaction and two reported answer transitions.

Run with Python 3.12 and matplotlib 3.11.1. PDFs retain vector text and graphics;
PNG previews are also generated. No document-level data are required.
"""
from math import isclose
from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle

from regenerate import CORPORA, NAMES, read, unique, verify_data


OUT = Path(__file__).resolve().parents[1] / "figures"
INK = "#292b28"
OLIVE = "#646a4e"
CHARCOAL = "#343631"
PALE = "#f1f2ec"
RULE = "#a5a69d"
MM_PER_INCH = 25.4


def canvas(width, height):
    fig = Figure(figsize=(width / MM_PER_INCH, height / MM_PER_INCH), facecolor="white")
    FigureCanvasAgg(fig)
    return fig


def save(fig, name, title):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", metadata={"Title": title, "CreationDate": None,
                                               "ModDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=350)


def framework():
    width, height = 154, 78
    fig = canvas(width, height)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.set_axis_off()

    def box(x, y, w, h, fill="white", border=RULE):
        ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=border, lw=.55))

    def label(x, y, s, size=7.35, bold=False, ha="left", color=INK):
        ax.text(x, y, s, fontsize=size, fontweight="bold" if bold else "normal",
                color=color, ha=ha, va="center", linespacing=1.20)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6,
                                     linewidth=.7, color=OLIVE, shrinkA=0, shrinkB=0))

    def line(x1, x2, y, color=RULE):
        ax.plot([x1, x2], [y, y], lw=.55, color=color)

    # Page selection precedes any change in answering text. Retrieval is repeated
    # between raster levels in Study 1 and fixed across arms within each level.
    stages = [
        (.5, 33.2, "Source documents", "JDocQA / MP-DocVQA"),
        (38.6, 31.2, "Primary raster", "Degrade each page"),
        (74.7, 33.8, "OCR-based retrieval", "Tesseract + TF–IDF"),
        (113.4, 40.1, "Freeze up to two pages", "Same pages and order"),
    ]
    for x, w, title, subtitle in stages:
        box(x, .6, w, 13.5, PALE)
        label(x+w/2, 4.8, title, bold=True, ha="center")
        label(x+w/2, 10, subtitle, size=7.0, ha="center")
    for first, second in zip(stages, stages[1:]):
        arrow((first[0]+first[1]+.6, 7.35), (second[0]-.6, 7.35))

    box(.5, 17, 153, 7.5, "#ffffff")
    label(77, 20.75,
          "Same retained text within each pair (8,000-token cap)   •   Same page bytes across image arms",
          size=7.55, ha="center")
    arrow((133.4, 14.1), (133.4, 16.8))
    for center in (37.5, 116):
        arrow((center, 24.5), (center, 28.1))

    # The common matrix is shown once per study. Image-only is a separate control,
    # and InternVL appears only in the qualified, edited-text pair.
    for x, w, title, subtitle, baseline1, baseline2, outcome, followup in (
        (.5, 74.9, "Study 1   Changing the OCR engine",
         "100 documents per corpus", "Tesseract (T)", "PP-OCRv6 (P)",
         r"ANLS      $G_P$  and  $G_P-G_T$", "Qwen image-only control"),
        (78.6, 74.9, "Study 2   Reader-qualified edited text",
         "600 candidates across both corpora", "Observed OCR (O)", "Edited text (R)",
         r"Correctness      $G_O-G_R$", "Qwen image-only control   •   InternVL on R pair"),
    ):
        box(x, 28.5, w, 49)
        box(x, 28.5, w, 10.6, PALE)
        label(x+2.5, 31.8, title, size=8.6, bold=True, color=OLIVE)
        label(x+2.5, 36.4, subtitle, size=7.15)
        if baseline1.startswith("Tesseract"):
            label(x+2.5, 43.9, "Read fixed pages with each engine.", size=7.45)
        else:
            label(x+2.5, 42.4, "Edit before question → allocate O and R", size=7.1)
            label(x+2.5, 46.2, "Screen on retained R → model QA on 372", size=7.1, bold=True)
        label(x+2.5, 51, "Qwen input", size=7.15, bold=True)
        for offset, text in ((43.0, "Text only"), (62.5, "Text + image")):
            label(x+offset, 51, text, size=7.15, bold=True, ha="center")
        line(x+2.5, x+w-2.5, 53.3)
        for y, name in ((56.6, baseline1), (61.5, baseline2)):
            label(x+2.5, y, name)
            for offset in (43, 62.5):
                label(x+offset, y, "✓", size=9.6, ha="center", color=INK)
        label(x+2.5, 66.8, followup, size=7.10)
        line(x+.1, x+w-.1, 70)
        label(x+2.5, 73.9, outcome, size=8.0, bold=True)
    save(fig, "framework", "Controlled page inputs and paired comparisons in the two studies")


def interaction():
    arms, contrasts = read("earlier_ocr_arms"), read("earlier_ocr_contrasts")
    fig = canvas(74, 48)
    panels = fig.subplots(1, 2, sharey=True)
    fig.subplots_adjust(left=.165, right=.982, top=.755, bottom=.17, wspace=.23)
    for ax, corpus in zip(panels, CORPORA):
        means = {}
        for arm in ("T1", "P1", "T2", "P2"):
            row = unique(arms, corpus=corpus, level="r4", arm=arm)
            if int(row["size"]) != 100:
                raise ValueError(f"Unexpected document count for {corpus}/{arm}")
            means[arm] = float(row["mean_anls_points"])
        g_t, g_p = means["T2"] - means["T1"], means["P2"] - means["P1"]
        for name, value in (("I", g_p-g_t), ("G_P", g_p)):
            row = unique(contrasts, corpus=corpus, population="prespecified", contrast=name)
            if not isclose(value, float(row["estimate_anls_points"]), abs_tol=1e-10):
                raise ValueError(f"Arm means disagree with primary contrast {corpus}/{name}")
        ax.plot([0, 1], [means["T1"], means["P1"]], marker="o", color=OLIVE,
                markersize=4.1, lw=1.45, label="Text only")
        ax.plot([0, 1], [means["T2"], means["P2"]], marker="s", color=CHARCOAL,
                markerfacecolor="white", markersize=4.05, lw=1.4, ls="--", label="Text + image")
        ax.set(xlim=(-.46, 1.46), ylim=(0, 80), yticks=[0, 20, 40, 60, 80])
        ax.set_xticks([0, 1], ["Tesseract", "PP-OCRv6"])
        ax.set_title(NAMES[corpus], fontsize=8.1, weight="bold", pad=3.7)
        ax.grid(axis="y", color="#dedfd8", lw=.53)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set(color=RULE, linewidth=.65)
        ax.tick_params(axis="both", labelsize=7.05, pad=2.4, length=0)
    panels[0].set_ylabel("Mean ANLS (points)", fontsize=7.3, labelpad=1.7)
    handles, labels = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.555, 1.02), fontsize=7.3, handlelength=2,
               handletextpad=.5, columnspacing=1.4)
    save(fig, "interaction", "Study 1: OCR engine and image input on the degraded raster")


def examples():
    """Draw the two published cases using paraphrases of the retained evidence."""
    fig = canvas(154, 36)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 154), ylim=(36, 0))
    ax.set_axis_off()
    cases = [
        ("(a) Correction on MP-DocVQA", "Printed page number", "Page number 33 was retained.",
         "33", "1", "33"),
        ("(b) New error on JDocQA", "Policyholder’s share", "The stated coverage limit was 90%.",
         "10%", "10%", "100%"),
    ]
    for left, (title, task, evidence, reference, before, after) in zip((0.6, 79.0), cases):
        def label(x, y, value, size=7.15, bold=False, align="left"):
            ax.text(left+x, y, value, va="center", ha=align, fontsize=size,
                    color=INK, weight="bold" if bold else "normal")

        ax.plot([left, left+74], [.5, .5], color=OLIVE, lw=1.0)
        label(.8, 3.3, title, size=8.3, bold=True)
        label(.8, 8.2, "Task")
        label(16, 8.2, task)
        label(.8, 12.2, "Evidence")
        label(16, 12.2, evidence)
        label(.8, 16.2, "Reference")
        label(16, 16.2, reference, bold=True)
        ax.add_patch(Rectangle((left, 19.2), 74, 16.3, facecolor=PALE, edgecolor="none"))
        for x, arm, value in ((17.8, "Text only", before), (56.2, "Text + image", after)):
            label(x, 22.0, arm, align="center")
            label(x, 27.3, value, size=10.7, bold=True, align="center")
            label(x, 32.5, "Correct" if value == reference else "Incorrect", align="center")
        ax.add_patch(FancyArrowPatch((left+30, 27.3), (left+44, 27.3), arrowstyle="-|>",
                                     color=CHARCOAL, lw=.8, mutation_scale=8))
    save(fig, "examples", "Two paired answer transitions on reader-qualified documents")


if __name__ == "__main__":
    verify_data(None)
    with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 7.35, "pdf.fonttype": 42,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "mathtext.fontset": "dejavusans"}):
        framework()
        interaction()
        examples()
