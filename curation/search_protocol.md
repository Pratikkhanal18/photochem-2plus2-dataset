# Literature search and curation protocol for extending the dataset

Goal: at least 500 records from at least 60 primary papers, with at least 20% of records below 50% yield,
so that models can be validated on many independent sources.

## 1. Where to search

| Database | Suggested query (adapt to the interface) |
|---|---|
| Reaxys | Reaction search: product contains a cyclobutane formed from two C=C bonds (draw the [2+2] reaction with a mapped four-membered ring); filter conditions to *irradiation*, *hν*, *light*, *photo*; publication years 2000–present. |
| SciFinder-n | Reaction search with the same drawn transformation; refine by "photochemical" and by catalyst role. |
| Web of Science / Google Scholar | Keyword combinations: "[2+2] photocycloaddition", "energy transfer catalysis cyclobutane", "visible light [2+2]", "enantioselective [2+2] photocycloaddition", "triplet sensitization cyclobutane", "intramolecular [2+2] enone". |
| Reviews | Use recent reviews only to find primary papers; take the data from the primary paper and its SI. |

Record each query, the database, the date run and the number of hits in a search log, since the paper's
Methods section must report them.

## 2. Inclusion criteria

Include a reaction if all of these hold:

- It forms a cyclobutane (or cyclobutene) by [2+2] cycloaddition under light.
- Reactant and product structures are fully specified in the paper or SI.
- A quantitative outcome is reported (yield, conversion, or ratio).
- Conditions (light source or wavelength, solvent, temperature) are stated.

Exclude: thermal or metal-mediated non-photochemical [2+2]; reactions reported only as "trace" with no
structure; duplicate reports of the same experiment in a later paper (keep the original).

## 3. Priorities that improve the models most

1. **Optimization tables in the SI.** These contain failed and low-yield conditions that literature datasets
   otherwise lack. Enter every row, including "no reaction" as 0% yield.
2. **Many papers, not many rows from one paper.** Prefer 5–15 records from each of many papers over
   50 records from one, because the benchmark tests transfer between sources.
3. **Underrepresented catalyst classes.** Organic sensitizers (thioxanthones, cyanoarenes such as 4CzIPN,
   Eosin Y), Ir complexes, and visible-light Lewis-acid systems are thin or absent in version 1.0.
4. **Reported selectivity.** Records with a diastereomeric ratio and the identity of the major isomer.

## 4. Quality control

- Draw structures and export SMILES; do not type them. Run `full_descriptor_analysis.py`, which writes
  `smiles_parse_log.csv`; fix every record that fails to parse.
- Every row is checked against the paper by a second person (`Checked_by`).
- Type ratios only into the text-formatted `dr_text` / `er_text` columns.
- Before merging, check for duplicates (same substrates, catalyst, conditions and source).

## 5. Merging into the dataset

Convert template columns to the existing schema (`Yield_pct` ÷ 100 → `Yield`, `Primary_DOI` → `Source_DOI`,
and so on), append to the dataset, re-run both pipelines, and release as a new Zenodo version (1.1.0).
Keep version 1.0.0 unchanged so that the paper's results remain reproducible.
