"""
Main script for protein classification experiments
"""
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from dataset import ProteinDataLoader
from model import get_classifiers
from utils import (
    save_cv_fold_indices, evaluate_model_cv, print_results_summary, 
    save_results, create_run_directory, save_all_models
)


def main(args):
    """Run protein classification experiments"""
    print("\nLOADING AND PREPROCESSING DATA")
    print("-" * 70)
    
    # Convert paths to Path objects for cross-platform compatibility
    train_data_path = Path(args.train_data)
    test_data_path = Path(args.test_data) if args.test_data else None
    
    data_loader = ProteinDataLoader(
        data_path=str(train_data_path),
        label_col=args.label_col,
        id_col=args.id_col,
        random_state=args.random_state
    )
    
    X_train_raw, y_train, X_test, y_test, train_df = data_loader.get_train_test_split(
        train_path=str(train_data_path),
        test_path=str(test_data_path) if test_data_path else None
    )

    print(f"\nSETTING UP {args.n_folds}-FOLD CROSS-VALIDATION")
    print("-" * 70)
    
    cv_splitter = StratifiedKFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=args.random_state
    )
    
    data_folder = train_data_path.parent

    # Generate fold_df here, but save it later if --save-models is true
    fold_df = save_cv_fold_indices(
        X=X_train_raw,
        y=y_train,
        df=train_df,
        cv_splitter=cv_splitter,
        output_path=data_folder / "temp_cv_fold_indices.csv",  # Save temporarily, will be moved
        id_col=args.id_col
    )
    
    print(f"   Number of folds: {args.n_folds}")
    print(f"   Samples per fold: ~{len(X_train_raw) // args.n_folds}")
    print(f"   Per-fold preprocessing: Each fold fits its own scaler")
    
    print(f"\nEVALUATING CLASSIFIERS")
    print("-" * 70)
    
    classifiers = get_classifiers(
        random_state=args.random_state,
        nn_patience=args.nn_patience,
        transformer_patience=args.transformer_patience,
    )
    results = []
    detailed_results = {}
    test_available = X_test is not None

    for clf_name, clf in classifiers.items():
        print(f"\n  {clf_name}...")

        try:
            result = evaluate_model_cv(
                clf=clf,
                X_train_raw=X_train_raw,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv_splitter=cv_splitter,
                clf_name=clf_name
            )

            results.append(result)
            detailed_results[clf_name] = result

            # Key CV metrics (balanced accuracy preferred over accuracy)
            print(f"     CV-AUC: {result['cv_auc_mean']:.3f} ± {result['cv_auc_std']:.3f}")
            print(f"     CV-BalAcc: {result.get('cv_bal_acc_mean', float('nan')):.3f} ± {result.get('cv_bal_acc_std', float('nan')):.3f}")
            print(f"     CV-F1: {result.get('cv_f1_mean', float('nan')):.3f} ± {result.get('cv_f1_std', float('nan')):.3f}")
            
            if test_available:
                print(f"     Test-AUC: {result['test_auc_mean']:.3f} ± {result['test_auc_std']:.3f}")
                print(f"     Test-BalAcc: {result.get('test_bal_acc_mean', float('nan')):.3f} ± {result.get('test_bal_acc_std', float('nan')):.3f}")
                print(f"     Test-F1: {result.get('test_f1_mean', float('nan')):.3f} ± {result.get('test_f1_std', float('nan')):.3f}")

        except Exception as e:
            print(f"     Error: {str(e)[:50]}...")
            continue

    results_df = pd.DataFrame(results)
    print_results_summary(results_df, test_available=test_available)
    
    print(f"\nSAVING RESULTS")
    print("-" * 70)
    
    if args.save_models:
        # Save everything to run directory (centralized)
        run_dir = create_run_directory()
        
        # Save CV fold indices to run directory
        cv_splits_run_path = run_dir / "cv_fold_indices.csv"
        fold_df.to_csv(cv_splits_run_path, index=False)
        print(f"Saved CV fold indices to: {cv_splits_run_path}")
        
        # Save results to run directory
        save_results(detailed_results, run_dir)
        results_csv_path = run_dir / "cv_results_summary.csv"
        results_df.to_csv(results_csv_path, index=False)
        print(f"Saved CV results summary to: {results_csv_path}")
        
        # Save models and final test results
        save_all_models(
            classifiers=classifiers,
            results_df=results_df,
            X_train_raw=X_train_raw,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            run_dir=run_dir
        )
    else:
        # Fallback: save to data folder for quick runs without models
        save_results(detailed_results, data_folder)
        results_csv_path = data_folder / "classifier_results_summary.csv"
        results_df.to_csv(results_csv_path, index=False)
        print(f"Saved results summary to: {results_csv_path}")
    
    print("\nEXPERIMENT COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein classification with multiple models")
    
    parser.add_argument("--train-data", type=str, default="src/data/protein/proteomic_encoder_train.csv", help="Path to training data CSV")
    parser.add_argument("--test-data", type=str, default="src/data/protein/proteomic_encoder_test.csv", help="Path to test data CSV (optional)")
    parser.add_argument("--label-col", type=str, default="research_group", help="Name of label column")
    parser.add_argument("--id-col", type=str, default="RID", help="Name of subject ID column")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-models", action="store_true", help="Train and save all models on full training set to timestamped run directory")
    parser.add_argument("--nn-patience", type=int, default=20, help="Early stopping patience for neural network")
    parser.add_argument("--transformer-patience", type=int, default=10, help="Early stopping patience for transformer")
    
    args = parser.parse_args()
    main(args)