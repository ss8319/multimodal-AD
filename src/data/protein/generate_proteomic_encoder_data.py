import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create proteomic_train_set.csv by removing RIDs present in merged_proteomic_mri_mprage.csv"
    )
    parser.add_argument(
        "--csv1",
        default="src/data/protein/proteomic_w_labels.csv",
        help="Path to proteomic_w_labels.csv (must contain column RID)",
    )
    parser.add_argument(
        "--csv2",
        default="src/data/protein/merged_proteomic_mri_mprage.csv",
        help="Path to merged_proteomic_mri_mprage.csv (must contain column RID)",
    )
    parser.add_argument(
        "--out",
        default="src/data/protein/proteomic_train_set.csv",
        help="Output path for the filtered CSV",
    )
    args = parser.parse_args()

    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    if "RID" not in df1.columns:
        raise SystemExit(f"{args.csv1} must contain column 'RID'. Found: {list(df1.columns)}")
    if "RID" not in df2.columns:
        raise SystemExit(f"{args.csv2} must contain column 'RID'. Found: {list(df2.columns)}")

    rids_to_exclude = set(df2["RID"].astype(str).str.strip())
    result = df1[~df1["RID"].astype(str).str.strip().isin(rids_to_exclude)].copy()

    result.to_csv(args.out, index=False)
    print(
        f"Wrote {args.out} with {len(result)} rows (from {len(df1)}; excluded {len(rids_to_exclude)} unique RIDs)"
    )


if __name__ == "__main__":
    main()


