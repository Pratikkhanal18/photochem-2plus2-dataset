# Photochemical [2+2] Cycloaddition Dataset with Traced Provenance and a Study-Grouped Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23838653.svg)](https://doi.org/10.5281/zenodo.23838653)

Curated dataset of 249 photochemical [2+2] cycloaddition records: 220 extracted from the review of
Poplata, Tröster, Zou and Bach (*Chem. Rev.* 2016, 116, 9748; https://doi.org/10.1021/acs.chemrev.5b00723)
and 29 from Hörmann *et al.* (*Angew. Chem. Int. Ed.* 2020, 59, 9659; https://doi.org/10.1002/anie.202001634).
248 of the 249 records are traced to the primary papers the review cites for them (139 independent studies,
272 primary-paper DOIs). The code reproduces every statistic, model result and figure in the accompanying paper.

**Authors:** Pratik Khanal and Bashu Dev Neupane (Crimson College of Technology, Butwal, Nepal); Amrit Khanal (Kathmandu Multiple College, Gaushala, Kathmandu, Nepal)

**Paper:** [journal, year and DOI to be added on publication]

**Dataset DOI:** [10.5281/zenodo.23838653](https://doi.org/10.5281/zenodo.23838653)

## Contents

| Path | Description |
|---|---|
| `data/ML_merged_cleaned.xlsx` | Original spreadsheet (249 records, 20 columns), unchanged. It contains spreadsheet artifacts (see below); use the cleaned CSV or apply `corrections.csv`. |
| `data/ML_cleaned_dataset.csv` | Cleaned records: corrections applied, numeric yield, wavelength and temperature, harmonized solvent, catalyst class, decoded selectivity ratios, and provenance fields (`Review_scheme`, `Primary_DOIs`, `Study_ID`, `Provenance_confidence`). |
| `data/provenance.csv` | For every record: source type, review scheme, cited review reference numbers, primary-paper DOIs and first authors, how the assignment was made, confidence level and `Study_ID`. |
| `data/corrections.csv` | Every correction made to the original spreadsheet, with original value, corrected value and reason. Both pipelines apply this file first. |
| `data/data_dictionary.csv` | Definition, unit and origin of every field in the cleaned CSV. |
| `data/fold_assignments.csv` | Exact study-grouped cross-validation folds used in the paper. |
| `data/fold_assignments_dictionary.csv` | Explanation of the fold columns. |
| `code/analysis_pipeline.py` | Baseline pipeline: corrections, ratio repair, statistics, figures 1–3, baseline yield models, selectivity classifier, bootstrap CIs, fold export. |
| `code/robustness.py` | Label-shuffling control, learning curve and leave-one-catalyst-class-out validation (run after the baseline pipeline). |
| `code/full_descriptor_analysis.py` | Descriptor pipeline: RDKit descriptors, Morgan and reaction fingerprints, optional Mordred, XGBoost and SHAP, with the same validation protocol. |
| `code/make_fig4.py`, `code/make_fig6.py` | Redraw paper Figs 4 and 6 from the result files. |
| `code/requirements.txt` | Python dependencies. |
| `results_descriptor_run/` | Complete output of the descriptor run reported in the paper, including the SMILES parse log and `environment.json` with exact software versions. |
| `curation/curation_template.xlsx`, `curation/search_protocol.md` | Template and protocol for adding records from new primary studies. |

## Quick start

```bash
pip install -r code/requirements.txt
cp data/ML_merged_cleaned.xlsx .
python code/analysis_pipeline.py                       # baseline results and Figs 1-3
python code/robustness.py && python code/make_fig6.py  # robustness analyses and Fig. 6
python code/full_descriptor_analysis.py --out results  # descriptor results (needs RDKit, XGBoost, SHAP)
```

The scripts look for `corrections.csv` and `provenance.csv` in the working directory or in `data/`.
The baseline pipeline was run with Python 3.12, pandas 3.0, NumPy 2.5, SciPy 1.18, scikit-learn 1.9 and
matplotlib 3.11; the descriptor run records its versions in `results/environment.json`.

## Provenance

Records from the review were linked to primary papers through the review's schemes: for each scheme, the
references cited in the sentence that refers to it (or, failing that, the nearest citation) were taken from
the review's open-access full text and checked manually where ambiguous. Records without a scheme number were
matched to schemes by structure and conditions. Records sharing any cited reference form one study
(`Study_ID`). Each assignment has a confidence level (high 164, medium 83, low 1, unassigned 1).

## Spreadsheet artifacts that were corrected

- **Ratios stored as times.** Ratios entered as `a:b` (e.g. `91:9`) were stored by Excel as durations
  (`91 h 09 min`, shown as `3 days, 19:09:00`). They are recovered exactly (minutes ÷ 60 = `a`, remainder = `b`).
  One ratio with decimals (`97.5:2.5`) stayed as text and is parsed directly: 76 usable diastereomeric ratios.
- **Drag-fill increments.** Copying cells downwards incremented their last number: the review DOI
  (…5b00723 became …5b00724 to …5b00746 in 170 records), reaction times (22 records, set to missing),
  ring-closure digits of catalyst and solvent SMILES (8 records, giving invalid SMILES), and four wavelengths.
  All are listed in `data/corrections.csv`. Three catalyst loadings (records 149–151) are flagged but unchanged.

When editing the data in a spreadsheet, format DOI and ratio columns as **Text** and do not use drag-fill.

## Validation protocol

All models are reported under two schemes:

- **Random:** 5-fold cross-validation repeated with five seeds (ten for classification).
- **Study-grouped:** all records of one study are kept in the same fold (yield: 5 balanced folds, 5 repeats;
  selectivity: leave-one-study-out over 27 studies).

Random splitting overstates performance on new chemistry. Please report study-grouped results when comparing
with this benchmark, using `data/fold_assignments.csv`.

## Licence

- Data (`data/`, `curation/`): Creative Commons Attribution 4.0 International (CC BY 4.0), see `LICENSE-DATA`.
- Code (`code/`): MIT, see `LICENSE`.

The reaction data are factual information extracted from the cited publications; please also cite the review
and the primary papers listed in `data/provenance.csv`.

## Citation

See `CITATION.cff`.
