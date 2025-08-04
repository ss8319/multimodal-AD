#!/usr/bin/env python3
"""
Simple script to interpret the 8 autoencoder features
"""

import os
import sys
from protein_feature_interpretation import (
    analyze_feature_contributions,
    plot_feature_interpretation,
    create_feature_summary_report,
    analyze_feature_correlations_with_diagnosis
)

def main():
    """Run feature interpretation on your trained autoencoder"""
    
    # Find the most recent experiment directory
    base_path = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM"
    
    # Look for experiment directories within dataset folders
    if os.path.exists(base_path):
        # Find all dataset folders
        dataset_folders = []
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                dataset_folders.append(item_path)
        
        # Find experiments within each dataset folder
        all_experiments = []
        for dataset_folder in dataset_folders:
            if os.path.exists(dataset_folder):
                for exp_item in os.listdir(dataset_folder):
                    exp_path = os.path.join(dataset_folder, exp_item)
                    if os.path.isdir(exp_path):
                        all_experiments.append(exp_path)
        
        if all_experiments:
            # Sort by creation time and use the most recent
            all_experiments.sort(key=lambda x: os.path.getctime(x), reverse=True)
            experiment_dir = all_experiments[0]
            print(f"Using most recent experiment: {experiment_dir}")
        else:
            print("No experiment directories found!")
            return
    else:
        print(f"Base path not found: {base_path}")
        return
    
    try:
        print("Analyzing feature contributions...")
        feature_analysis, protein_cols = analyze_feature_contributions(experiment_dir)
        
        print("Creating feature interpretation plots...")
        plot_feature_interpretation(feature_analysis, 
                                   os.path.join(experiment_dir, 'feature_interpretation.png'))
        
        print("Creating feature summary report...")
        create_feature_summary_report(feature_analysis, experiment_dir,
                                     os.path.join(experiment_dir, 'feature_interpretation_report.txt'))
        
        print("Analyzing feature correlations with diagnosis...")
        correlations = analyze_feature_correlations_with_diagnosis(experiment_dir)
        
        print("\n" + "="*60)
        print("FEATURE INTERPRETATION COMPLETED!")
        print("="*60)
        print(f"Results saved to: {experiment_dir}")
        print("\nFiles created:")
        print("- feature_interpretation.png: Visual breakdown of each feature")
        print("- feature_interpretation_report.txt: Detailed text report")
        print("- feature_diagnosis_correlations.png: Feature vs AD diagnosis correlations")
        
        print("\nQUICK SUMMARY:")
        print("Each of your 8 features is a weighted combination of original proteins.")
        print("Positive weights = proteins that increase the feature")
        print("Negative weights = proteins that decrease the feature")
        print("\nThe most important features for AD diagnosis are those with highest")
        print("absolute correlation with the AD/CN labels.")
        
    except Exception as e:
        print(f"Error during feature interpretation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    