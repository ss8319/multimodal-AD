🗂️ 1. CSFMRM_CONSOLIDATED_DATA.xlsx
✅ Description:
A cleaned, quality-controlled, and consolidated version of the data.

Likely includes:

Only peptides that passed QC (e.g. low CVs, good detection)

Normalized and log₂-transformed values

Well-annotated sample and peptide metadata

Possibly summary statistics (mean, std, etc.)

🧠 Use Case:
Ready for downstream statistical analysis, modeling, and biomarker discovery.

Reflects what the MRM Data Primer calls the "QC-filtered processed peptide-level data".

🧾 2. CSFMRM.csv
✅ Description:
Likely the raw or minimally processed peptide intensity matrix.

May include:

All measured peptides (including those that failed QC)

Raw light/heavy ratios or untransformed intensity values

Possibly more missing data or peptides flagged for exclusion in downstream steps

🧠 Use Case:
For users who want to re-apply QC, normalization, or examine excluded peptides.

Matches what the primer calls the “raw Skyline export” or raw output before filtering.