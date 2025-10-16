"""
Split proteomic_encoder_data.csv into train/test sets with configurable demographic stratification
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_age_bins(df, n_bins=4):
    """Create age bins with better distribution"""
    try:
        # Try quantile-based binning first
        df['age_bin'] = pd.qcut(df['subject_age'], q=n_bins, labels=False, duplicates='drop')
        return df
    except ValueError:
        # Fallback to equal-width binning if quantiles fail
        df['age_bin'] = pd.cut(df['subject_age'], bins=n_bins, labels=False, include_lowest=True)
        return df


def apply_stratification_strategy(df, stratify_by, strategy, test_size, random_state, age_bins, min_samples_per_group):
    """Apply different stratification strategies"""
    
    if strategy == 'strict':
        return strict_stratification(df, stratify_by, test_size, random_state, age_bins, min_samples_per_group)
    elif strategy == 'relaxed':
        return relaxed_stratification(df, stratify_by, test_size, random_state, age_bins, min_samples_per_group)
    elif strategy == 'hierarchical':
        return hierarchical_stratification(df, stratify_by, test_size, random_state, age_bins, min_samples_per_group)
    elif strategy == 'balanced':
        return balanced_stratification(df, stratify_by, test_size, random_state, age_bins, min_samples_per_group)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def strict_stratification(df, stratify_by, test_size, random_state, age_bins, min_samples_per_group):
    """Strict stratification - fail if impossible"""
    df = df.copy()
    
    # Create stratification key
    strat_components = []
    
    if 'diagnosis' in stratify_by:
        strat_components.append(df['research_group'].astype(str))
    
    if 'age' in stratify_by:
        df = create_age_bins(df, age_bins)
        strat_components.append(df['age_bin'].astype(str))
    
    if 'sex' in stratify_by:
        strat_components.append(df['Sex'].astype(str))
    
    # Create combined stratification key
    if len(strat_components) == 1:
        df['strat_key'] = strat_components[0]
    else:
        df['strat_key'] = strat_components[0]
        for component in strat_components[1:]:
            df['strat_key'] = df['strat_key'] + '_' + component
    
    # Check for groups with insufficient samples
    strat_counts = df['strat_key'].value_counts()
    insufficient_groups = strat_counts[strat_counts < min_samples_per_group].index.tolist()
    
    if insufficient_groups:
        raise ValueError(f"Strict stratification failed: {len(insufficient_groups)} groups have < {min_samples_per_group} samples: {insufficient_groups}")
    
    # Perform stratified split
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['strat_key'], random_state=random_state
    )
    
    return train_df, test_df


def relaxed_stratification(df, stratify_by, test_size, random_state, age_bins, min_samples_per_group):
    """Relaxed stratification - fallback to fewer factors if needed"""
    df = df.copy()
    
    # Try with all factors first, then progressively remove factors
    for num_factors in range(len(stratify_by), 0, -1):
        try:
            current_factors = stratify_by[:num_factors]
            print(f"  Trying stratification with {num_factors} factors: {current_factors}")
            
            strat_components = []
            
            if 'diagnosis' in current_factors:
                strat_components.append(df['research_group'].astype(str))
            
            if 'age' in current_factors:
                df = create_age_bins(df, age_bins)
                strat_components.append(df['age_bin'].astype(str))
            
            if 'sex' in current_factors:
                strat_components.append(df['Sex'].astype(str))
            
            # Create stratification key
            if len(strat_components) == 1:
                df['strat_key'] = strat_components[0]
            else:
                df['strat_key'] = strat_components[0]
                for component in strat_components[1:]:
                    df['strat_key'] = df['strat_key'] + '_' + component
            
            # Check if stratification is possible
            strat_counts = df['strat_key'].value_counts()
            insufficient_groups = strat_counts[strat_counts < min_samples_per_group]
            
            if len(insufficient_groups) == 0:
                print(f"  ✓ Success with {num_factors} factors")
                break
            else:
                print(f"  ✗ Failed with {num_factors} factors: {len(insufficient_groups)} insufficient groups")
                continue
                
        except Exception as e:
            print(f"  ✗ Failed with {num_factors} factors: {e}")
            continue
    
    # Handle insufficient groups by assigning them to training
    strat_counts = df['strat_key'].value_counts()
    insufficient_groups = strat_counts[strat_counts < min_samples_per_group].index.tolist()
    
    if insufficient_groups:
        print(f"  Assigning {len(insufficient_groups)} insufficient groups to training set")
        insufficient_mask = df['strat_key'].isin(insufficient_groups)
        insufficient_samples = df[insufficient_mask].copy()
        remaining_df = df[~insufficient_mask].copy()
        
        if len(remaining_df) > 0:
            train_df, test_df = train_test_split(
                remaining_df, test_size=test_size, stratify=remaining_df['strat_key'], random_state=random_state
            )
            train_df = pd.concat([train_df, insufficient_samples], ignore_index=True)
        else:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df['strat_key'], random_state=random_state
        )
    
    return train_df, test_df


def hierarchical_stratification(df, stratify_by, test_size, random_state, age_bins, min_samples_per_group):
    """Hierarchical stratification - prioritize most important factors"""
    df = df.copy()
    
    # Priority order: diagnosis > age > sex
    priority_order = ['diagnosis', 'age', 'sex']
    available_factors = [f for f in priority_order if f in stratify_by]
    
    print(f"  Hierarchical order: {available_factors}")
    
    # Start with highest priority factor
    current_factors = [available_factors[0]]
    
    for i in range(1, len(available_factors) + 1):
        try:
            current_factors = available_factors[:i]
            print(f"  Trying with factors: {current_factors}")
            
            strat_components = []
            
            if 'diagnosis' in current_factors:
                strat_components.append(df['research_group'].astype(str))
            
            if 'age' in current_factors:
                df = create_age_bins(df, age_bins)
                strat_components.append(df['age_bin'].astype(str))
            
            if 'sex' in current_factors:
                strat_components.append(df['Sex'].astype(str))
            
            # Create stratification key
            if len(strat_components) == 1:
                df['strat_key'] = strat_components[0]
            else:
                df['strat_key'] = strat_components[0]
                for component in strat_components[1:]:
                    df['strat_key'] = df['strat_key'] + '_' + component
            
            # Check if this level works
            strat_counts = df['strat_key'].value_counts()
            insufficient_groups = strat_counts[strat_counts < min_samples_per_group]
            
            if len(insufficient_groups) == 0:
                print(f"  ✓ Success with {len(current_factors)} factors")
                break
            else:
                print(f"  ✗ {len(insufficient_groups)} insufficient groups, trying fewer factors")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            break
    
    # Use the best stratification we found
    strat_counts = df['strat_key'].value_counts()
    insufficient_groups = strat_counts[strat_counts < min_samples_per_group].index.tolist()
    
    if insufficient_groups:
        print(f"  Assigning {len(insufficient_groups)} insufficient groups to training set")
        insufficient_mask = df['strat_key'].isin(insufficient_groups)
        insufficient_samples = df[insufficient_mask].copy()
        remaining_df = df[~insufficient_mask].copy()
        
        if len(remaining_df) > 0:
            train_df, test_df = train_test_split(
                remaining_df, test_size=test_size, stratify=remaining_df['strat_key'], random_state=random_state
            )
            train_df = pd.concat([train_df, insufficient_samples], ignore_index=True)
        else:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df['strat_key'], random_state=random_state
        )
    
    return train_df, test_df


def balanced_stratification(df, stratify_by, test_size, random_state, age_bins, min_samples_per_group):
    """Balanced stratification - optimize for best balance across all factors"""
    df = df.copy()
    
    # For age + diagnosis, use adaptive age binning
    if 'age' in stratify_by and 'diagnosis' in stratify_by:
        return age_diagnosis_balanced_stratification(df, test_size, random_state, age_bins, min_samples_per_group)
    
    # Fallback to hierarchical for other combinations
    return hierarchical_stratification(df, stratify_by, test_size, random_state, age_bins, min_samples_per_group)


def age_diagnosis_balanced_stratification(df, test_size, random_state, age_bins, min_samples_per_group):
    """Specialized stratification for age + diagnosis combination"""
    df = df.copy()
    
    # Try different age binning strategies
    best_split = None
    best_score = float('inf')
    
    for bin_strategy in ['quantile', 'equal_width', 'adaptive']:
        try:
            print(f"  Trying {bin_strategy} age binning...")
            
            if bin_strategy == 'quantile':
                df['age_bin'] = pd.qcut(df['subject_age'], q=age_bins, labels=False, duplicates='drop')
            elif bin_strategy == 'equal_width':
                df['age_bin'] = pd.cut(df['subject_age'], bins=age_bins, labels=False, include_lowest=True)
            elif bin_strategy == 'adaptive':
                # Adaptive binning: ensure each diagnosis has samples in each age bin
                df['age_bin'] = adaptive_age_binning(df, age_bins)
            
            # Create stratification key
            df['strat_key'] = df['research_group'].astype(str) + '_' + df['age_bin'].astype(str)
            
            # Check group sizes
            strat_counts = df['strat_key'].value_counts()
            insufficient_groups = strat_counts[strat_counts < min_samples_per_group]
            
            if len(insufficient_groups) == 0:
                # Perfect stratification possible
                train_df, test_df = train_test_split(
                    df, test_size=test_size, stratify=df['strat_key'], random_state=random_state
                )
                
                # Calculate balance score (lower is better)
                score = calculate_balance_score(train_df, test_df)
                print(f"    ✓ Perfect stratification, balance score: {score:.3f}")
                
                if score < best_score:
                    best_score = score
                    best_split = (train_df.copy(), test_df.copy())
            else:
                print(f"    ✗ {len(insufficient_groups)} insufficient groups")
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue
    
    if best_split is not None:
        return best_split
    else:
        # Fallback: use relaxed stratification
        print("  Falling back to relaxed stratification...")
        return relaxed_stratification(df, ['diagnosis', 'age'], test_size, random_state, age_bins, min_samples_per_group)


def adaptive_age_binning(df, n_bins):
    """Create age bins that ensure each diagnosis has samples in each bin"""
    # Start with quantile-based binning
    try:
        age_bins = pd.qcut(df['subject_age'], q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        age_bins = pd.cut(df['subject_age'], bins=n_bins, labels=False, include_lowest=True)
    
    # Check if each diagnosis has samples in each age bin
    diagnosis_counts = df.groupby(['research_group', age_bins]).size().unstack(fill_value=0)
    
    # If any diagnosis-age combination is empty, adjust bins
    if diagnosis_counts.min().min() == 0:
        # Use fewer bins
        for bins in range(n_bins-1, 1, -1):
            try:
                age_bins = pd.qcut(df['subject_age'], q=bins, labels=False, duplicates='drop')
                diagnosis_counts = df.groupby(['research_group', age_bins]).size().unstack(fill_value=0)
                if diagnosis_counts.min().min() > 0:
                    break
            except ValueError:
                continue
    
    return age_bins


def calculate_balance_score(train_df, test_df):
    """Calculate balance score (lower is better)"""
    score = 0
    
    # Age balance
    age_diff = abs(train_df['subject_age'].mean() - test_df['subject_age'].mean())
    score += age_diff
    
    # Diagnosis balance
    train_diag = train_df['research_group'].value_counts(normalize=True)
    test_diag = test_df['research_group'].value_counts(normalize=True)
    for diag in train_diag.index:
        if diag in test_diag.index:
            score += abs(train_diag[diag] - test_diag[diag])
    
    # Sex balance (if present)
    if 'Sex' in train_df.columns:
        train_sex = train_df['Sex'].value_counts(normalize=True)
        test_sex = test_df['Sex'].value_counts(normalize=True)
        for sex in train_sex.index:
            if sex in test_sex.index:
                score += abs(train_sex[sex] - test_sex[sex])
    
    return score


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
        '--stratification-strategy',
        choices=['strict', 'relaxed', 'hierarchical', 'balanced'],
        default='balanced',
        help='Stratification strategy: strict (fail if impossible), relaxed (fallback to fewer factors), hierarchical (prioritize factors), balanced (optimize balance)'
    )
    
    parser.add_argument(
        '--age-bins',
        type=int,
        default=4,
        help='Number of age bins for stratification (default: 4)'
    )
    
    parser.add_argument(
        '--min-samples-per-group',
        type=int,
        default=2,
        help='Minimum samples required per stratification group (default: 2)'
    )
    
    parser.add_argument(
        '--input-path',
        type=str,
        default='src/data/protein/proteomic_with_demographics.csv',
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
    
    # 2. Improved stratified split with multiple strategies
    print(f"\n{'=' * 70}")
    print("CREATING TRAIN/TEST SPLIT")
    print("=" * 70)
    print(f"Strategy: {args.stratification_strategy}")
    print(f"Min samples per group: {args.min_samples_per_group}")
    
    # Apply stratification strategy
    train_df, test_df = apply_stratification_strategy(
        df, args.stratify_by, args.stratification_strategy, 
        args.test_size, args.random_state, args.age_bins, args.min_samples_per_group
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
