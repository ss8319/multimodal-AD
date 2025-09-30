import csv
import re
import shutil
import os
from pathlib import Path
from collections import defaultdict, Counter

def extract_rid_from_subject(subject_id):
    """
    Extract RID from Subject ID format like '067_S_0029' -> 29
    """
    if not subject_id:
        return None
    
    # Extract the number after the last underscore
    match = re.search(r'_(\d+)$', str(subject_id))
    if match:
        return int(match.group(1))
    return None

def normalize_viscode(viscode):
    """
    Normalize VISCODE/Visit codes for matching
    """
    if not viscode:
        return None
    
    viscode = str(viscode).lower().strip()
    
    # Common mappings
    visit_mappings = {
        'bl': 'bl',
        'baseline': 'bl',
        'm00': 'bl',
        'sc': 'sc',
        'screening': 'sc',
        'm06': 'm06',
        'm12': 'm12',
        'm18': 'm18',
        'm24': 'm24',
        'm36': 'm36',
        'm48': 'm48',
        'm60': 'm60',
        'm72': 'm72',
        'm84': 'm84',
        'm96': 'm96',
        'm108': 'm108',
        'm120': 'm120'
    }
    
    # Check for exact matches first
    if viscode in visit_mappings:
        return visit_mappings[viscode]
    
    # Check for pattern matches (e.g., '4_init', '4_m12')
    # Note: Removed automatic '_init' -> 'bl' mapping as it's incorrect
    # '4_init' should remain as '4_init', not be converted to 'bl'
    
    # Extract month patterns (e.g., 'm12', '12m', '12_month')
    # But be more specific to avoid false matches like '4_init'
    month_patterns = [
        r'^m(\d+)$',      # m12, m24, etc.
        r'^(\d+)m$',      # 12m, 24m, etc.
        r'^(\d+)_month$'  # 12_month, etc.
    ]
    
    for pattern in month_patterns:
        month_match = re.search(pattern, viscode)
        if month_match:
            month_num = int(month_match.group(1))
            if month_num == 0:
                return 'bl'
            else:
                return f'm{month_num:02d}'
    
    # If no pattern matches, return as-is (e.g., '4_init' stays '4_init')
    return viscode

def load_proteomic_data(csv_path):
    """Load proteomic data from CSV"""
    print(f"📂 Loading proteomic data: {csv_path}")
    
    proteomic_data = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            proteomic_data.append(row)
    
    print(f"📊 Proteomic data: {len(proteomic_data)} records")
    return proteomic_data

def load_mri_data_mprage(csv_path):
    """Load MRI data and filter for MPRAGE scans only"""
    print(f"📂 Loading MRI data: {csv_path}")
    
    mri_data = []
    mri_mprage = []
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            mri_data.append(row)
            # Filter for MPRAGE scans (including MP-RAGE variant)
            description = row.get('Description', '').lower()
            if 'mprage' in description or 'mp-rage' in description:
                mri_mprage.append(row)
    
    print(f"📊 Total MRI data: {len(mri_data)} records")
    print(f"📊 MPRAGE scans: {len(mri_mprage)} records")
    return mri_data, mri_mprage

def match_proteomic_mri_mprage(csv1_path, csv2_path, output_path=None):
    """
    Match proteomic data with MPRAGE MRI data
    """
    print(" Loading datasets...")
    print("=" * 60)
    
    # Load data
    proteomic_data = load_proteomic_data(csv1_path)
    mri_data, mri_mprage = load_mri_data_mprage(csv2_path)
    
    # Create lookup dictionaries
    print("\n🔄 Creating proteomic lookup dictionary...")
    proteomic_lookup = {}
    for i, row in enumerate(proteomic_data):
        rid = int(row['RID']) if row['RID'].isdigit() else None
        viscode_norm = normalize_viscode(row['VISCODE'])
        if rid and viscode_norm:
            key = (rid, viscode_norm)
            if key in proteomic_lookup:
                print(f"  [{i+1:2d}] WARNING  DUPLICATE KEY: RID {rid}, VISCODE '{row['VISCODE']}' -> '{viscode_norm}' (overwriting previous)")
            proteomic_lookup[key] = row
            print(f"  [{i+1:2d}] RID {rid}, VISCODE '{row['VISCODE']}' -> '{viscode_norm}' ✓")
        else:
            print(f"  [{i+1:2d}] RID {rid}, VISCODE '{row['VISCODE']}' -> '{viscode_norm}' ✗ (skipped)")
    
    print(f"\n📊 Proteomic lookup keys created: {len(proteomic_lookup)}")
    
    # Match with MPRAGE data
    matched_records = []
    matched_keys = set()
    match_count = 0
    
    print("\n Matching datasets on RID and Visit (MPRAGE only)...")
    print("=" * 60)
    
    for i, mri_row in enumerate(mri_mprage):
        subject = mri_row['Subject']
        visit = mri_row['Visit']
        description = mri_row['Description']
        
        # Extract RID from subject
        rid_extracted = extract_rid_from_subject(subject)
        visit_norm = normalize_viscode(visit)
        
        if rid_extracted and visit_norm:
            key = (rid_extracted, visit_norm)
            if key in proteomic_lookup:
                # Create merged record
                proteomic_row = proteomic_lookup[key]
                merged_row = {**proteomic_row, **mri_row}
                matched_records.append(merged_row)
                matched_keys.add(key)
                match_count += 1
                print(f"  [{match_count:2d}] ✓ MATCH: RID {rid_extracted}, Visit '{visit}' -> '{visit_norm}' | {subject} | {description}")
            else:
                if i < 10:  # Show first 10 non-matches for debugging
                    print(f"  [--] ✗ NO MATCH: RID {rid_extracted}, Visit '{visit}' -> '{visit_norm}' | {subject}")
        else:
            if i < 10:  # Show first 10 invalid records for debugging
                print(f"  [--] ✗ INVALID: Subject '{subject}', Visit '{visit}' | RID: {rid_extracted}, Visit_norm: {visit_norm}")
    
    print(f"\nOK Successfully matched {len(matched_records)} records")
    
    # Identify missing subjects from proteomic data
    print(f"\n Analyzing missing subjects from proteomic data...")
    print("=" * 60)
    matched_rids = set()
    for key in matched_keys:
        matched_rids.add(key[0])  # key[0] is RID
    
    proteomic_rids = set()
    for row in proteomic_data:
        proteomic_rids.add(int(row['RID']))
    
    missing_rids = proteomic_rids - matched_rids
    print(f"Missing subjects: {len(missing_rids)} out of {len(proteomic_rids)}")
    
    if missing_rids:
        print(f"📋 Missing RIDs that need MPRAGE scans:")
        # Create lookup for efficiency (avoid O(n²) nested loop)
        rid_to_proteomic = {int(row['RID']): row for row in proteomic_data}
        
        for rid in sorted(missing_rids):
            if rid in rid_to_proteomic:
                row = rid_to_proteomic[rid]
                print(f"  RID {rid}: VISCODE '{row['VISCODE']}' -> '{normalize_viscode(row['VISCODE'])}' | Group: {row.get('research_group', 'Unknown')}")
            else:
                print(f"  RID {rid}: WARNING  Not found in proteomic data")
    
    # Check for multiple MPRAGE scans per subject
    print("\n Checking for multiple MPRAGE scans per subject...")
    print("=" * 60)
    rid_counts = Counter()
    for record in matched_records:
        rid = int(record['RID'])
        rid_counts[rid] += 1
    
    multiple_scans = {rid: count for rid, count in rid_counts.items() if count > 1}
    
    if multiple_scans:
        print(f"WARNING  Found {len(multiple_scans)} subjects with multiple MPRAGE scans:")
        for rid, count in list(multiple_scans.items())[:10]:
            print(f"  RID {rid}: {count} MPRAGE scans")
        
        # Select the first MPRAGE scan for each subject
        print("\n🔄 Selecting first MPRAGE scan per subject...")
        print("=" * 60)
        seen_rids = set()
        deduplicated_records = []
        
        for i, record in enumerate(matched_records):
            rid = int(record['RID'])
            if rid not in seen_rids:
                deduplicated_records.append(record)
                seen_rids.add(rid)
                print(f"  [{len(deduplicated_records):2d}] ✓ KEPT: RID {rid} | {record['Subject']} | {record['Description']}")
            else:
                print(f"  [--] ✗ SKIPPED: RID {rid} (duplicate) | {record['Subject']} | {record['Description']}")
        
        matched_records = deduplicated_records
        print(f"\n📊 After deduplication: {len(matched_records)} records")
    else:
        print("OK All subjects have exactly one MPRAGE scan")
    
    # Display statistics
    print("\n📊 MERGE STATISTICS")
    print("=" * 60)
    print(f"Original proteomic records: {len(proteomic_data)}")
    print(f"Original MRI records: {len(mri_data)}")
    print(f"MPRAGE MRI records: {len(mri_mprage)}")
    print(f"Final matched records: {len(matched_records)}")
    print(f"Proteomic match rate: {len(matched_records)/len(proteomic_data)*100:.1f}%")
    print(f"MPRAGE match rate: {len(matched_records)/len(mri_mprage)*100:.1f}%")
    
    # Count unique subjects
    unique_rids = set()
    for record in matched_records:
        unique_rids.add(int(record['RID']))
    
    print(f"Unique subjects in merged data: {len(unique_rids)}")
    
    # Group distribution
    group_counts = Counter()
    for record in matched_records:
        group = record.get('research_group', 'Unknown')
        group_counts[group] += 1
    
    print(f"\n📋 Group Distribution:")
    for group, count in group_counts.items():
        print(f"  {group}: {count} subjects")
    
    # Sample of merged data
    print("\n Sample merged data:")
    print("=" * 60)
    if matched_records:
        sample_record = matched_records[0]
        sample_cols = ['RID', 'VISCODE', 'Subject', 'Visit', 'research_group', 'Group', 'Description']
        for col in sample_cols:
            if col in sample_record:
                print(f"  {col}: {sample_record[col]}")
    
    # Save merged data
    if output_path and matched_records:
        print(f"\nSAVED: Saving merged data to: {output_path}")
        print("=" * 60)
        
        # Get all unique column names
        all_columns = set()
        for record in matched_records:
            all_columns.update(record.keys())
        
        # Organize columns: metadata first, then proteins
        print("🔄 Organizing columns: metadata first, then proteins...")
        
        # Define metadata columns (order matters)
        metadata_columns = [
            # Core identifiers
            'RID', 'Subject', 'VISCODE', 'Visit', 'VISCODE_norm', 'Visit_norm',
            # Subject info
            'research_group', 'Group', 'Sex', 'Age', 'subject_age',
            # MRI scan info
            'Image Data ID', 'Description', 'Type', 'Modality', 'Format',
            'Acq Date', 'Downloaded', 'MRI_acquired',
            # Additional extracted fields
            'RID_extracted'
        ]
        
        # Find protein columns (anything that contains '_' and looks like a protein)
        protein_columns = []
        other_columns = []
        
        for col in all_columns:
            if col not in metadata_columns:
                # Protein columns typically have format like "PROTEIN_PEPTIDE" 
                if '_' in col and col.upper() == col:  # All caps with underscore
                    protein_columns.append(col)
                else:
                    other_columns.append(col)
        
        # Sort protein columns alphabetically for consistency
        protein_columns.sort()
        other_columns.sort()
        
        # Final column order: metadata -> proteins -> others
        ordered_columns = []
        
        # Add metadata columns that exist in data
        for col in metadata_columns:
            if col in all_columns:
                ordered_columns.append(col)
        
        # Add other non-protein columns
        ordered_columns.extend(other_columns)
        
        # Add protein columns at the end
        ordered_columns.extend(protein_columns)
        
        print(f"📊 Column organization:")
        print(f"  - Metadata columns: {len([c for c in metadata_columns if c in all_columns])}")
        print(f"  - Other columns: {len(other_columns)}")
        print(f"  - Protein columns: {len(protein_columns)}")
        print(f"  - Total columns: {len(ordered_columns)}")
        
        print(f"📊 Writing {len(matched_records)} records with organized columns...")
        
        with open(output_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=ordered_columns)
            writer.writeheader()
            writer.writerows(matched_records)
        
        print("OK Merged data saved successfully!")
        print(f"FOLDER: File saved: {output_path}")
        print("📋 Column order: Metadata → Other → Proteins (alphabetical)")
    
    return matched_records

def count_unique_rids_mprage(csv_file_path):
    """
    Count unique RID values in the MPRAGE-filtered merged CSV file
    """
    print(f"\n Analyzing unique RIDs in MPRAGE-filtered dataset: {csv_file_path}")
    print("=" * 60)
    
    rids = []
    total_records = 0
    
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            total_records += 1
            rid = row.get('RID', '').strip()
            if rid and rid.isdigit():
                rids.append(int(rid))
    
    unique_rids = set(rids)
    rid_counts = Counter(rids)
    
    print(f"📊 ANALYSIS RESULTS:")
    print(f"Total records: {total_records}")
    print(f"Unique RID values: {len(unique_rids)}")
    
    # Show RID distribution
    print(f"\n📋 RID Distribution:")
    for rid in sorted(unique_rids):
        count = rid_counts[rid]
        print(f"  RID {rid}: {count} record(s)")
    
    # Group distribution
    group_counts = Counter()
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            group = row.get('research_group', 'Unknown')
            group_counts[group] += 1
    
    print(f"\n📋 Group Distribution:")
    for group, count in group_counts.items():
        print(f"  {group}: {count} subjects")
    
    return {
        'total_records': total_records,
        'unique_rids': len(unique_rids),
        'rid_counts': rid_counts
    }

def copy_matched_mri_images_simple(csv_file_path, source_root_dir, target_root_dir):
    """
    Copy MRI image folders for matched subjects based on CSV data
    
    Args:
        csv_file_path: Path to the merged CSV file
        source_root_dir: Root directory where ADNI images are stored
        target_root_dir: Target directory for copied images
        
    Returns:
        dict: Statistics about the copy operation
    """
    print(f"\nFOLDER: Copying matched MRI images...")
    print("=" * 60)
    print(f"Reading from CSV: {csv_file_path}")
    print(f"Source root: {source_root_dir}")
    print(f"Target root: {target_root_dir}")
    
    # Create target directory
    target_path = Path(target_root_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_root_dir)
    if not source_path.exists():
        print(f"ERROR Source directory not found: {source_root_dir}")
        return {'error': 'Source directory not found'}
    
    stats = {
        'subjects_processed': 0,
        'subjects_found': 0,
        'subjects_not_found': 0,
        'folders_copied': 0,
        'folders_skipped': 0,
        'errors': 0
    }
    
    # Read the CSV file
    print(f"\n🔄 Reading CSV file...")
    matched_subjects = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            matched_subjects.append({
                'subject': row['Subject'],
                'description': row['Description'], 
                'image_data_id': row['Image Data ID'],
                'rid': row['RID']
            })
    
    print(f"📊 Found {len(matched_subjects)} subjects to process")
    
    # Process each subject
    for i, subject_info in enumerate(matched_subjects, 1):
        subject_id = subject_info['subject']
        description = subject_info['description'].strip()  # Remove leading/trailing spaces
        image_data_id = subject_info['image_data_id']
        rid = subject_info['rid']
        
        print(f"\n[{i:2d}/{len(matched_subjects)}] Processing RID {rid} ({subject_id})")
        print(f"  Description: '{description}'")
        print(f"  Image Data ID: {image_data_id}")
        
        stats['subjects_processed'] += 1
        
        try:
            # Simplified approach: search for Image Data ID directly under subject directory
            subject_dir = source_path / "ADNI" / subject_id
            
            if not subject_dir.exists():
                print(f"  ERROR Subject directory not found: {subject_dir}")
                stats['subjects_not_found'] += 1
                continue
            
            # Search for Image Data ID anywhere under the subject directory
            print(f"   Searching for Image Data ID: {image_data_id}")
            image_dirs = list(subject_dir.rglob(image_data_id))
            
            if not image_dirs:
                print(f"  ERROR Image Data ID directory not found: {image_data_id}")
                # List available directories for debugging
                print("   Available scan directories:")
                for scan_dir in subject_dir.iterdir():
                    if scan_dir.is_dir():
                        print(f"    - {scan_dir.name}")
                        # Look for any Image Data ID directories in this scan
                        for subdir in scan_dir.rglob("I*"):
                            if subdir.is_dir() and subdir.name.startswith("I"):
                                print(f"      └─ {subdir.name}")
                stats['subjects_not_found'] += 1
                continue
            
            # Use the first matching image directory
            source_image_dir = image_dirs[0]
            print(f"  OK Found image dir: {source_image_dir}")
            
            # Verify this matches our expected description (for logging purposes)
            actual_description = source_image_dir.parent.parent.name  # Go up two levels to get scan type
            if description.lower().replace(' ', '') in actual_description.lower().replace('_', '').replace('-', ''):
                print(f"  OK Description matches: '{actual_description}' ~ '{description}'")
            else:
                print(f"  WARNING  Description mismatch: found '{actual_description}', expected '{description}'")
            
            # Build target path maintaining the same structure
            # Extract the relative path from ADNI onwards
            relative_path = source_image_dir.relative_to(source_path / "ADNI")
            target_image_dir = target_path / relative_path
            
            print(f"  FOLDER: Target: {target_image_dir}")
            
            # Check if target already exists
            if target_image_dir.exists():
                print(f"  WARNING  Target directory already exists, skipping...")
                stats['folders_skipped'] += 1
                continue
            
            # Copy the entire directory
            print(f"  🔄 Copying directory...")
            shutil.copytree(source_image_dir, target_image_dir)
            
            # Count files copied
            files_in_dir = len(list(target_image_dir.rglob("*")))
            print(f"  OK Copied directory with {files_in_dir} items")
            
            stats['subjects_found'] += 1
            stats['folders_copied'] += 1
            
        except Exception as e:
            print(f"  ERROR Error processing {subject_id}: {e}")
            stats['errors'] += 1
    
    # Print summary
    print(f"\n📊 COPY OPERATION SUMMARY")
    print("=" * 60)
    print(f"Subjects processed: {stats['subjects_processed']}")
    print(f"Subjects with images found: {stats['subjects_found']}")
    print(f"Subjects with no images: {stats['subjects_not_found']}")
    print(f"Folders copied: {stats['folders_copied']}")
    print(f"Folders skipped (existing): {stats['folders_skipped']}")
    print(f"Errors encountered: {stats['errors']}")
    print(f"Target directory: {target_root_dir}")
    
    return stats

if __name__ == "__main__":
    # File paths
    csv1_path = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\proteomic_mri_with_labels.csv"
    csv2_path = r"C:\Users\User\github_repos\AD_CN_all_available_data_final\AD_CN_all_available_data.csv"
    output_path = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\merged_proteomic_mri_mprage.csv"
    
    # Match the datasets (MPRAGE only)
    merged_data = match_proteomic_mri_mprage(csv1_path, csv2_path, output_path)
    
    # Count unique RIDs
    stats = count_unique_rids_mprage(output_path)

    # Copy matched MRI images to organized folder
    source_root_dir = r"C:\Users\User\github_repos\AD_CN_all_available_data_final"
    target_root_dir = r"C:\Users\User\github_repos\AD_CN_MRI_final"
    
    print(f"\n🔄 Starting MRI image copy operation...")
    copy_stats = copy_matched_mri_images_simple(output_path, source_root_dir, target_root_dir)
    
    print(f"\nOK Data matching and image copying completed!")
    print(f"📊 Final merged dataset: {len(merged_data)} records")
    print(f"ANSWER: ANSWER: There are {stats['unique_rids']} unique RID fields in the MPRAGE-filtered dataset")
    print(f"SAVED: CSV saved to: {output_path}")
    print(f"FOLDER: Images copied to: {target_root_dir}")
    if 'error' not in copy_stats:
        print(f"IMAGES:  Image folders copied: {copy_stats['folders_copied']}/{stats['unique_rids']}")
