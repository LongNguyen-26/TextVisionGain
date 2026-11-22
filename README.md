# TextVisionGain

Reproduce the three tables and four figures in *How the Text Baseline Changes the Value of Page Images in Document Question Answering*, by Viet Nhat Long Nguyen and Nguyen Nang Hung Van. The public inputs are six aggregate CSVs, the study design and two reported example transitions. Exhibit generation runs on a CPU without downloading either dataset.

## Reproduce the exhibits

Clone this repository and run the commands below from its root in a Python 3.12 environment. Matplotlib and NumPy are pinned to the versions used for the figures.

```sh
python -m pip install -r requirements.txt
python scripts/regenerate.py
python scripts/presentation_figures.py
```

Both scripts create their output directories on first use. Re-running either script replaces its own exhibits.

| Exhibit | Inputs | Output |
|---|---|---|
| Table I, OCR-engine contrasts | earlier_ocr_arms.csv, earlier_ocr_contrasts.csv | tables/engine.tex |
| Table II, reader-qualified Qwen estimates | main_secondary.csv (Qwen arm bounds), main_primary.csv (S1/A contrasts) | tables/primary.tex |
| Table III, correction and harm counts | main_opportunities.csv, edited-text rows | tables/flips.tex |
| Figure 1, study design | design labels in presentation_figures.py | figures/framework.pdf |
| Figure 2, within-page engine comparison | earlier_ocr_arms.csv, earlier_ocr_contrasts.csv | figures/interaction.pdf |
| Figure 3, screening by length | main_flow.csv | figures/selection.pdf |
| Figure 4, correction and new error | two paraphrased examples in presentation_figures.py | figures/examples.pdf |

The plotting script also writes PNG previews of Figures 1, 2 and 4. To check a reproduction, compare the table entries, plotted values and labels. PDF metadata and font versions can affect file hashes.

## Scope and access

The scripts verify all six CSVs against `data/SHA256SUMS` and use the archived estimates and interval endpoints. Recomputing the paired estimates requires linked document-level outcomes, strata and judgments from the controlled archive. Repeating inference also requires access to the source documents and model inputs. Figure 4 reproduces the two published transitions. The underlying source documents, edited transcripts, individual grading records and prediction logs are held separately. [Methods, scoring and dataset access](METHODS.md)

## License and citation

The MIT license covers the original Python scripts. The aggregate CSVs are supplied for reproduction and verification of the manuscript. The code license grants no rights to third-party documents, annotations or model weights. Obtain source data from the providers under their respective terms. To cite this software, use [CITATION.cff](CITATION.cff). Cite the original dataset papers when using their data.
