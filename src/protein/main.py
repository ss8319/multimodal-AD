"""
Main script for protein classification experiments
"""
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from dataset import ProteinDataLoader
from model import get_classifiers
from utils import save_cv_fold_indices, evaluate_model_cv, print_results_summary, save_results


def main(args):
    """Run protein classification experiments"""
    # ==================================================
    # 1. DATA LOADING AND PREPROCESSING
    # ==================================================
    print("\n📂 LOADING AND PREPROCESSING DATA")
    print("-" * 70)
    
    data_loader = ProteinDataLoader(
        data_path=args.train_data,
        label_col=args.label_col,
        id_col=args.id_col,
        random_state=args.random_state
    )
    
    X_train, y_train, X_test, y_test, train_df = data_loader.get_train_test_split(
        train_path=args.train_data,
        test_path=args.test_data
    )

    # ==================================================
    # 2. SETUP CROSS-VALIDATION
    # ==================================================
    print(f"\n🔄 SETTING UP {args.n_folds}-FOLD CROSS-VALIDATION")
    print("-" * 70)
    
    cv_splitter = StratifiedKFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=args.random_state
    )
    
    # Save CV fold indices for reproducibility
    data_folder = Path(args.train_data).parent
    cv_splits_path = data_folder / "cv_fold_indices.csv"
    
    fold_df = save_cv_fold_indices(
        X=X_train,
        y=y_train,
        df=train_df,
        cv_splitter=cv_splitter,
        output_path=cv_splits_path,
        id_col=args.id_col
    )
    
    print(f"   • Number of folds: {args.n_folds}")
    print(f"   • Samples per fold: ~{len(X_train) // args.n_folds}")
    
    # # ==================================================
    # # 3. MODEL EVALUATION
    # # ==================================================
    print(f"\n🤖 EVALUATING CLASSIFIERS")
    print("-" * 70)
    
    classifiers = get_classifiers(random_state=args.random_state)
    results = []
    detailed_results = {}
    
    test_available = X_test is not None
    
    for clf_name, clf in classifiers.items():
        print(f"\n🔄 {clf_name}...")
        
        try:
            # Evaluate model with CV and optional test set
            result = evaluate_model_cv(
                clf=clf,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv_splitter=cv_splitter,
                clf_name=clf_name
            )
            
            results.append(result)
            detailed_results[clf_name] = result
            
            # Print results
            print(f"   ✅ CV-AUC: {result['cv_auc_mean']:.3f} ± {result['cv_auc_std']:.3f}")
            print(f"   ✅ CV-Acc: {result['cv_acc_mean']:.3f} ± {result['cv_acc_std']:.3f}")
            
            if test_available:
                print(f"   🎯 Test-AUC: {result['test_auc_mean']:.3f} ± {result['test_auc_std']:.3f}")
                print(f"   🎯 Test-Acc: {result['test_acc_mean']:.3f} ± {result['test_acc_std']:.3f}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}...")
            continue
    
    # ==================================================
    # 4. RESULTS SUMMARY
    # ==================================================
    results_df = pd.DataFrame(results)
    print_results_summary(results_df, test_available=test_available)
    
    # ==================================================
    # 5. SAVE RESULTS
    # ==================================================
    print(f"\n💾 SAVING RESULTS")
    print("-" * 70)
    
    save_results(detailed_results, data_folder)
    
    # Save summary table
    results_csv_path = data_folder / "classifier_results_summary.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"💾 Saved results summary to: {results_csv_path}")
    
    print("\n✅ EXPERIMENT COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein classification with multiple models")
    
    # Data paths
    parser.add_argument(
        "--train-data",
        type=str,
        default="src/data/protein/proteomic_encoder_train.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="src/data/protein/proteomic_encoder_test.csv",
        help="Path to test data CSV (optional)"
    )
    
    # Data columns
    parser.add_argument(
        "--label-col",
        type=str,
        default="research_group",
        help="Name of label column"
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="RID",
        help="Name of subject ID column"
    )
    
    # Experiment settings
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    main(args)