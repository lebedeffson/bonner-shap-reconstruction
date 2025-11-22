import os
import glob
import json
import shutil

def clean_results():
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print("Results directory not found.")
        return

    keep_patterns = [
        '*vanilla_r2_09*',  # Best Vanilla
        '*two_stage_fixed_gamma*', # Best Two-Stage SHAP
        '*integrated_v5*', # Current Attempt
    ]
    
    # Files to keep
    keep_files = set()
    for pattern in keep_patterns:
        keep_files.update(glob.glob(os.path.join(results_dir, pattern)))
    
    # Always keep samples and predictions for kept models
    # (Already handled by glob patterns if naming is consistent)

    print(f"Keeping {len(keep_files)} files...")
    
    # All files
    all_files = glob.glob(os.path.join(results_dir, '*'))
    
    deleted_count = 0
    for file_path in all_files:
        if file_path not in keep_files:
            if os.path.isdir(file_path) and 'spectra' not in file_path: # Don't delete spectra folder
                 shutil.rmtree(file_path)
                 deleted_count += 1
            elif os.path.isfile(file_path):
                os.remove(file_path)
                deleted_count += 1
                
    print(f"Deleted {deleted_count} files/directories.")

if __name__ == "__main__":
    clean_results()

