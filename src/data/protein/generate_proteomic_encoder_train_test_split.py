"""
Split proteomic_encoder_data.csv into train/test sets with configurable demographic stratification
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Split proteomic data into train/test sets with configurable stratification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --stratify-by diagnosis age sex    # Stratify by all demographics
  %(prog)s --stratify-by diagnosis age       # Stratify by diagnosis and age only
  %(prog)s --stratify-by diagnosis           # Stratify by diagnosis only
  %(prog)s --stratify-by age sex             # Stratify by age and sex only
        """
    )
    
    parser.add_argument(
        '--stratify-by', 
        nargs='+', 
        choices=['diagnosis', 'age', 'sex'],
        default=['diagnosis', 'age', 'sex'],
        help='Demographic features to stratify by (default: all three)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.20,
        help='Proportion of data for test set (default: 0.20)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--input-path',
        type=str,
        default='multimodal-AD/src/data/protein/proteomic_with_demographics.csv',
        help='Path to input CSV file (default: src/data/protein/proteomic_with_demographics.csv)'
    )
    
    args = parser.parse_args()
    
    # Validate stratification choices
    if not args.stratify_by:
        raise ValueError("Must specify at least one stratification feature")
    
    # Paths
    input_path = Path(args.input_path)
    output_dir = input_path.parent
    train_path = output_dir / "proteomic_encoder_train.csv"
    test_path = output_dir / "proteomic_encoder_test.csv"
     
    # Load data
    df = pd.read_csv(input_path)
    print("=" * 70)
    print("PROTEOMICS ENCODER TRAIN/TEST SPLIT")
    print("=" * 70)
    print(f"\nLoaded {len(df)} samples from {input_path.name}")
    print(f"Stratification features: {', '.join(args.stratify_by)}")
    print(f"Test size: {args.test_size:.1%}")
    print(f"Random state: {args.random_state}")
    
    # 1. Calculate statistics
    print(f"\n{'=' * 70}")
    print("DATASET STATISTICS")
    print("=" * 70)
    
    age_mean = df['subject_age'].mean()
    age_std = df['subject_age'].std()
    print(f"\nAge Statistics:")
    print(f"  Mean: {age_mean:.2f} years")
    print(f"  Std:  {age_std:.2f} years")
    
    group_counts = df['research_group'].value_counts()
    print(f"\nDiagnosis Distribution:")
    for group, count in group_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {group}: {count} ({pct:.1f}%)")
    
    sex_counts = df['Sex'].value_counts()
    print(f"\nSex Distribution:")
    for sex, count in sex_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {sex}: {count} ({pct:.1f}%)")
    
    # 2. Stratified split by selected demographic features
    print(f"\n{'=' * 70}")
    print("CREATING TRAIN/TEST SPLIT")
    print("=" * 70)
    
    # Build stratification key based on selected features
    strat_components = []
    
    if 'diagnosis' in args.stratify_by:
        strat_components.append(df['research_group'].astype(str))
    
    if 'age' in args.stratify_by:
        # Create age bins for stratification (to balance age across splits)
        # Use quantile-based bins for better age distribution
        df['age_bin'] = pd.qcut(df['subject_age'], q=4, labels=False, duplicates='drop')
        strat_components.append(df['age_bin'].astype(str))
    
    if 'sex' in args.stratify_by:
        strat_components.append(df['Sex'].astype(str))
    
    # Create combined stratification key
    if len(strat_components) == 1:
        df['strat_key'] = strat_components[0]
    else:
        df['strat_key'] = strat_components[0]
        for component in strat_components[1:]:
            df['strat_key'] = df['strat_key'] + '_' + component
    
    # Check for single-sample groups and handle them
    strat_counts = df['strat_key'].value_counts()
    single_sample_groups = strat_counts[strat_counts == 1].index.tolist()
    
    if single_sample_groups:
        print(f"Warning: Found {len(single_sample_groups)} single-sample groups: {single_sample_groups}")
        print("These samples will be assigned to train set to enable splitting.")
        
        # Assign single-sample groups to train set
        single_sample_mask = df['strat_key'].isin(single_sample_groups)
        single_samples = df[single_sample_mask].copy()
        remaining_df = df[~single_sample_mask].copy()
        
        # Split remaining data
        if len(remaining_df) > 0:
            train_df, test_df = train_test_split(
                remaining_df,
                test_size=args.test_size,
                stratify=remaining_df['strat_key'],
                random_state=args.random_state
            )
            # Add single samples to train set
            train_df = pd.concat([train_df, single_samples], ignore_index=True)
        else:
            # Fallback: random split if stratification fails
            train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.random_state)
    else:
        # Normal stratified split
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            stratify=df['strat_key'],
            random_state=args.random_state
        )
    
    # Remove temporary columns (only if they exist)
    temp_cols = ['strat_key']
    if 'age' in args.stratify_by:
        temp_cols.append('age_bin')
    
    train_df = train_df.drop(columns=[col for col in temp_cols if col in train_df.columns])
    test_df = test_df.drop(columns=[col for col in temp_cols if col in test_df.columns])
    
    # 3. Show split statistics
    print(f"\nSplit Sizes:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Train set statistics
    train_age_mean = train_df['subject_age'].mean()
    train_age_std = train_df['subject_age'].std()
    train_group_counts = train_df['research_group'].value_counts()
    train_sex_counts = train_df['Sex'].value_counts()
    
    print(f"\nTrain Set:")
    print(f"  Age: {train_age_mean:.2f} ± {train_age_std:.2f} years")
    print(f"  Diagnosis:")
    for group, count in train_group_counts.items():
        pct = (count / len(train_df)) * 100
        print(f"    {group}: {count} ({pct:.1f}%)")
    print(f"  Sex:")
    for sex, count in train_sex_counts.items():
        pct = (count / len(train_df)) * 100
        print(f"    {sex}: {count} ({pct:.1f}%)")
    
    # Test set statistics
    test_age_mean = test_df['subject_age'].mean()
    test_age_std = test_df['subject_age'].std()
    test_group_counts = test_df['research_group'].value_counts()
    test_sex_counts = test_df['Sex'].value_counts()
    
    print(f"\nTest Set:")
    print(f"  Age: {test_age_mean:.2f} ± {test_age_std:.2f} years")
    print(f"  Diagnosis:")
    for group, count in test_group_counts.items():
        pct = (count / len(test_df)) * 100
        print(f"    {group}: {count} ({pct:.1f}%)")
    print(f"  Sex:")
    for sex, count in test_sex_counts.items():
        pct = (count / len(test_df)) * 100
        print(f"    {sex}: {count} ({pct:.1f}%)")
    
    # 4. Save files
    print(f"\n{'=' * 70}")
    print("SAVING FILES")
    print("=" * 70)
    
    train_df.to_csv(train_path, index=False)
    print(f"\nSaved train set: {train_path}")
    print(f"  {len(train_df)} samples")
    
    test_df.to_csv(test_path, index=False)
    print(f"\nSaved test set: {test_path}")
    print(f"  {len(test_df)} samples")
    
    # 5. Balance check (only for stratified features)
    print(f"\n{'=' * 70}")
    print("BALANCE VERIFICATION")
    print("=" * 70)
    
    # Check diagnosis balance preservation (if stratified by diagnosis)
    if 'diagnosis' in args.stratify_by:
        print(f"\nDiagnosis Balance (Train vs Test):")
        for group in group_counts.index:
            train_pct = (train_group_counts.get(group, 0) / len(train_df)) * 100
            test_pct = (test_group_counts.get(group, 0) / len(test_df)) * 100
            diff = abs(train_pct - test_pct)
            status = "OK" if diff < 2.0 else "WARNING"
            print(f"  {group}: Train={train_pct:.1f}%, Test={test_pct:.1f}%, Diff={diff:.1f}% [{status}]")
    
    # Check sex balance preservation (if stratified by sex)
    if 'sex' in args.stratify_by:
        print(f"\nSex Balance (Train vs Test):")
        for sex in sex_counts.index:
            train_pct = (train_sex_counts.get(sex, 0) / len(train_df)) * 100
            test_pct = (test_sex_counts.get(sex, 0) / len(test_df)) * 100
            diff = abs(train_pct - test_pct)
            status = "OK" if diff < 2.0 else "WARNING"
            print(f"  {sex}: Train={train_pct:.1f}%, Test={test_pct:.1f}%, Diff={diff:.1f}% [{status}]")
    
    # Check age balance (if stratified by age)
    if 'age' in args.stratify_by:
        age_mean_diff = abs(train_age_mean - test_age_mean)
        age_status = "OK" if age_mean_diff < 2.0 else "WARNING"
        print(f"\nAge Balance:")
        print(f"  Mean difference: {age_mean_diff:.2f} years [{age_status}]")
        print(f"  Original: {age_mean:.2f} ± {age_std:.2f}")
        print(f"  Train:    {train_age_mean:.2f} ± {train_age_std:.2f}")
        print(f"  Test:     {test_age_mean:.2f} ± {test_age_std:.2f}")
    
    # Show non-stratified features for reference
    non_stratified = []
    if 'diagnosis' not in args.stratify_by:
        non_stratified.append('diagnosis')
    if 'age' not in args.stratify_by:
        non_stratified.append('age')
    if 'sex' not in args.stratify_by:
        non_stratified.append('sex')
    
    if non_stratified:
        print(f"\nNon-stratified features: {', '.join(non_stratified)}")
        print("  (These may have different distributions between train/test)")
    
    print(f"\n{'=' * 70}")
    print("SPLIT COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
