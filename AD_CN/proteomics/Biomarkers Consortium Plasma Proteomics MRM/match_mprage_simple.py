import csv
import re
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
    print("🔍 Loading datasets...")
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
                print(f"  [{i+1:2d}] ⚠️  DUPLICATE KEY: RID {rid}, VISCODE '{row['VISCODE']}' -> '{viscode_norm}' (overwriting previous)")
            proteomic_lookup[key] = row
            print(f"  [{i+1:2d}] RID {rid}, VISCODE '{row['VISCODE']}' -> '{viscode_norm}' ✓")
        else:
            print(f"  [{i+1:2d}] RID {rid}, VISCODE '{row['VISCODE']}' -> '{viscode_norm}' ✗ (skipped)")
    
    print(f"\n📊 Proteomic lookup keys created: {len(proteomic_lookup)}")
    
    # Match with MPRAGE data
    matched_records = []
    matched_keys = set()
    match_count = 0
    
    print("\n🔗 Matching datasets on RID and Visit (MPRAGE only)...")
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
    
    print(f"\n✅ Successfully matched {len(matched_records)} records")
    
    # Identify missing subjects from proteomic data
    print(f"\n🔍 Analyzing missing subjects from proteomic data...")
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
                print(f"  RID {rid}: ⚠️  Not found in proteomic data")
    
    # Check for multiple MPRAGE scans per subject
    print("\n🔍 Checking for multiple MPRAGE scans per subject...")
    print("=" * 60)
    rid_counts = Counter()
    for record in matched_records:
        rid = int(record['RID'])
        rid_counts[rid] += 1
    
    multiple_scans = {rid: count for rid, count in rid_counts.items() if count > 1}
    
    if multiple_scans:
        print(f"⚠️  Found {len(multiple_scans)} subjects with multiple MPRAGE scans:")
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
        print("✅ All subjects have exactly one MPRAGE scan")
    
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
    print("\n🔍 Sample merged data:")
    print("=" * 60)
    if matched_records:
        sample_record = matched_records[0]
        sample_cols = ['RID', 'VISCODE', 'Subject', 'Visit', 'research_group', 'Group', 'Description']
        for col in sample_cols:
            if col in sample_record:
                print(f"  {col}: {sample_record[col]}")
    
    # Save merged data
    if output_path and matched_records:
        print(f"\n💾 Saving merged data to: {output_path}")
        print("=" * 60)
        
        # Get all unique column names
        all_columns = set()
        for record in matched_records:
            all_columns.update(record.keys())
        
        print(f"📊 Writing {len(matched_records)} records with {len(all_columns)} columns...")
        
        with open(output_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=sorted(all_columns))
            writer.writeheader()
            writer.writerows(matched_records)
        
        print("✅ Merged data saved successfully!")
        print(f"📁 File saved: {output_path}")
    
    return matched_records

def count_unique_rids_mprage(csv_file_path):
    """
    Count unique RID values in the MPRAGE-filtered merged CSV file
    """
    print(f"\n🔍 Analyzing unique RIDs in MPRAGE-filtered dataset: {csv_file_path}")
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

if __name__ == "__main__":
    # File paths
    csv1_path = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\proteomic_mri_with_labels.csv"
    csv2_path = r"C:\Users\User\github_repos\AD_CN_all_available_data_final\AD_CN_all_available_data.csv"
    output_path = r"D:\ADNI\AD_CN\proteomics\Biomarkers Consortium Plasma Proteomics MRM\merged_proteomic_mri_mprage.csv"
    
    # Match the datasets (MPRAGE only)
    merged_data = match_proteomic_mri_mprage(csv1_path, csv2_path, output_path)
    
    # Count unique RIDs
    stats = count_unique_rids_mprage(output_path)
    
    print(f"\n✅ Data matching completed!")
    print(f"📊 Final merged dataset: {len(merged_data)} records")
    print(f"🎯 ANSWER: There are {stats['unique_rids']} unique RID fields in the MPRAGE-filtered dataset")
    print(f"💾 Saved to: {output_path}")
