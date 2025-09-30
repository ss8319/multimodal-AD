# Protein Data Organisation

## File Organisation

raw/CSFMRM.csv
- Raw proteomic data we use for experiments
- Pass Quality Control
- 303 Subjects hence, 303 rows of data

proteomic_w_labels.csv
- From CSFMRM.csv we merge the data with their AD or CN labels
- 158 Subjects (~5 duplicates)

merged_proteomic_mri_mprage.csv
- Proteomic dataset and their respective MRI information with AD or CN Labels
- 38 subjects
- used to train fusion model with 6:2:2 split
- corresponding images are in "github_repos\AD_CN_MRI_final"

proteomic_encoder_data.csv
- Use this data to train the protein encoder model
- From proteomic_w_labels.csv remove subjects (based on RID) in merged_proteomic_mri_mprage.csv
- 153 - 38 subjects = 115 subjects
- Keep to train protein encoder model based on 8:1:1 split

bash
'''
uv run python src/data/protein/generate_proteomic_encoder_data.py \
  --csv1 'src\data\protein\proteomic_w_labels.csv' \
  --csv2 'src\data\protein\merged_proteomic_mri_mprage.csv' \
  --out 'src\data\protein\proteomic_encoder_data.csv'
'''
