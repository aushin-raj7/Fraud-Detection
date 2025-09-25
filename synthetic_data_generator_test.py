import pandas as pd
import numpy as np
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class MedicalDataGenerator:
    def __init__(self):
        # Define normal ranges for lab tests
        self.normal_ranges = {
            'Hemoglobin': {'male': (13.5, 17.5), 'female': (12.0, 15.5)},  # g/dL
            'Hematocrit': {'male': (41, 50), 'female': (36, 44)},  # %
            'WBC': (4.0, 11.0),  # ×10³/μL
            'Neutrophils': (50, 70),  # %
            'Lymphocytes': (20, 40),  # %
            'Platelets': (150, 450),  # ×10³/μL
            'ALT': (7, 56),  # U/L
            'AST': (10, 40),  # U/L
            'Bilirubin': (0.3, 1.2),  # mg/dL
            'TotalProtein': (6.0, 8.3),  # g/dL
            'Albumin': (3.5, 5.0),  # g/dL
            'HbA1c': (4.0, 5.6),  # % (normal)
            'FastingGlucose': (70, 99),  # mg/dL (normal)
            'Creatinine': {'male': (0.7, 1.3), 'female': (0.6, 1.1)},  # mg/dL
            'BUN': (7, 20),  # mg/dL
            'Sodium': (136, 145),  # mmol/L
            'Potassium': (3.5, 5.0),  # mmol/L
            'ALP': (44, 147),  # U/L
            'Cholesterol': (0, 200),  # mg/dL (desirable)
            'LDL': (0, 100),  # mg/dL (optimal)
            'HDL': {'male': (40, 60), 'female': (50, 70)},  # mg/dL
            'Triglycerides': (0, 150),  # mg/dL
            'VLDL': (5, 30),  # mg/dL
            'Eosinophils': (1, 4),  # %
            'RandomGlucose': (70, 140),  # mg/dL
        }
        
        # Define categorical options
        self.categorical_options = {
            'HBsAg': ['Positive', 'Negative'],
            'HBV_DNA': ['Positive', 'Negative'],
            'HAV_IgM': ['Positive', 'Negative'],
            'RPR': ['Positive', 'Negative'],
            'Treponemal': ['Positive', 'Negative'],
            'Urine_RBC': ['Present', 'Absent'],
            'Urine_Blood': ['Positive', 'Negative'],
            'Urine_Protein': ['None', 'Mild', 'Severe'],
            'ABO_BloodGrp': ['A', 'B', 'AB', 'O'],
            'Rh_Factor': ['Positive', 'Negative'],
            'Parasites': ['Present', 'Absent'],
            'SampleType': ['Serum', 'Urine', 'Whole Blood']
        }

    def generate_candidate_blood_type(self, candidate_id: str) -> Tuple[str, str]:
        """Generate consistent blood type for a candidate using deterministic hash"""
        # Use deterministic hash for consistent results across sessions
        hash_obj = hashlib.md5(candidate_id.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Use hash to select blood type consistently
        abo_index = hash_int % len(self.categorical_options['ABO_BloodGrp'])
        rh_index = (hash_int // 4) % len(self.categorical_options['Rh_Factor'])
        
        abo = self.categorical_options['ABO_BloodGrp'][abo_index]
        rh = self.categorical_options['Rh_Factor'][rh_index]
        
        return abo, rh

    def generate_normal_record(self, report_id: str, candidate_id: str, report_date: str) -> Dict:
        """Generate a medically consistent (Unaltered) record"""
        # Assign gender for range-dependent tests
        gender = random.choice(['male', 'female'])
        
        # Get consistent blood type for candidate
        abo_group, rh_factor = self.generate_candidate_blood_type(candidate_id)
        
        record = {
            'report_id': report_id,
            'candidate_id': candidate_id,
            'report_date': report_date,
            'Target_Label': 'Unaltered'
        }
        
        # Generate normal lab values
        for test, ranges in self.normal_ranges.items():
            if isinstance(ranges, dict) and gender in ranges:
                min_val, max_val = ranges[gender]
            elif isinstance(ranges, tuple):
                min_val, max_val = ranges
            else:
                min_val, max_val = ranges
            
            record[test] = round(np.random.uniform(min_val, max_val), 2)
        
        # Ensure medical consistency for normal records
        # HbA1c and glucose consistency
        if record['HbA1c'] <= 5.6:
            record['FastingGlucose'] = round(np.random.uniform(70, 99), 1)
        
        # Liver enzyme consistency
        if record['Bilirubin'] <= 1.2:
            record['ALT'] = min(record['ALT'], 50)
            record['AST'] = min(record['AST'], 35)
        
        # CBC differential should sum to ~100%
        neutrophils = record['Neutrophils']
        lymphocytes = record['Lymphocytes']
        eosinophils = record['Eosinophils']
        
        # Calculate current sum and normalize to 100%
        current_sum = neutrophils + lymphocytes + eosinophils
        if current_sum > 0:
            # Proportionally adjust to sum to 100%, leaving room for other cells (monocytes, basophils)
            target_sum = 95  # Leave 5% for other cell types
            factor = target_sum / current_sum
            record['Neutrophils'] = round(neutrophils * factor, 1)
            record['Lymphocytes'] = round(lymphocytes * factor, 1)
            record['Eosinophils'] = round(eosinophils * factor, 1)
        
        # Generate categorical values (mostly normal)
        for category, options in self.categorical_options.items():
            if category in ['ABO_BloodGrp', 'Rh_Factor']:
                continue  # Already set
            elif category in ['HBsAg', 'HBV_DNA', 'HAV_IgM', 'RPR', 'Treponemal', 'Parasites']:
                # Mostly negative for infectious diseases
                record[category] = 'Negative' if random.random() < 0.9 else 'Positive'
                if category == 'Parasites':
                    record[category] = 'Absent' if record[category] == 'Negative' else 'Present'
            elif category in ['Urine_RBC', 'Urine_Blood']:
                # Handle urine tests with some correlation
                if category == 'Urine_RBC':
                    record[category] = 'Present' if random.random() < 0.1 else 'Absent'
                else:  # Urine_Blood
                    # Correlate with Urine_RBC - if RBC present, blood likely positive
                    if 'Urine_RBC' in record and record['Urine_RBC'] == 'Present':
                        record[category] = 'Positive' if random.random() < 0.8 else 'Negative'
                    else:
                        record[category] = 'Positive' if random.random() < 0.05 else 'Negative'
            elif category == 'Urine_Protein':
                record[category] = random.choices(['None', 'Mild', 'Severe'], 
                                                weights=[0.8, 0.15, 0.05])[0]
            else:
                record[category] = random.choice(options)
        
        # Set blood type
        record['ABO_BloodGrp'] = abo_group
        record['Rh_Factor'] = rh_factor
        
        # Ensure serology consistency for normal records
        if record['RPR'] == 'Positive':
            record['Treponemal'] = 'Positive' if random.random() < 0.8 else 'Negative'
        else:
            record['Treponemal'] = 'Negative'
            
        if record['HBsAg'] == 'Positive':
            record['HBV_DNA'] = 'Positive' if random.random() < 0.7 else 'Negative'
        
        return record

    def generate_altered_record(self, report_id: str, candidate_id: str, report_date: str) -> Dict:
        """Generate a medically inconsistent (Altered) record"""
        # Start with a normal record
        record = self.generate_normal_record(report_id, candidate_id, report_date)
        record['Target_Label'] = 'Altered'
        
        # Introduce specific inconsistencies
        alteration_type = random.choice([
            'glucose_inconsistency', 
            'liver_inconsistency', 
            'serology_inconsistency',
            'hematology_inconsistency',
            'blood_type_inconsistency',
            'specimen_mismatch',
            'extreme_values'
        ])
        
        if alteration_type == 'glucose_inconsistency':
            # HbA1c normal but fasting glucose diabetic
            record['HbA1c'] = round(np.random.uniform(4.0, 5.6), 1)
            record['FastingGlucose'] = round(np.random.uniform(200, 350), 1)
            
        elif alteration_type == 'liver_inconsistency':
            # Normal bilirubin but very high liver enzymes
            record['Bilirubin'] = round(np.random.uniform(0.3, 1.2), 2)
            record['ALT'] = round(np.random.uniform(200, 500), 1)
            record['AST'] = round(np.random.uniform(180, 450), 1)
            
        elif alteration_type == 'serology_inconsistency':
            # RPR positive but confirmatory negative, or vice versa
            if random.random() < 0.5:
                record['RPR'] = 'Positive'
                record['Treponemal'] = 'Negative'
            else:
                record['HBsAg'] = 'Positive'
                record['HBV_DNA'] = 'Negative'
                
        elif alteration_type == 'hematology_inconsistency':
            # Severe thrombocytopenia with normal other CBC
            record['Platelets'] = round(np.random.uniform(10, 50), 1)
            record['Hemoglobin'] = round(np.random.uniform(13, 16), 1)  # Normal
            record['WBC'] = round(np.random.uniform(5, 10), 1)  # Normal
            
        elif alteration_type == 'blood_type_inconsistency':
            # Change both ABO and Rh for same candidate (impossible)
            different_abo = [t for t in self.categorical_options['ABO_BloodGrp'] 
                            if t != record['ABO_BloodGrp']]
            different_rh = [t for t in self.categorical_options['Rh_Factor'] 
                           if t != record['Rh_Factor']]
            
            record['ABO_BloodGrp'] = random.choice(different_abo)
            if random.random() < 0.5:  # 50% chance to also change Rh
                record['Rh_Factor'] = random.choice(different_rh)
            
        elif alteration_type == 'specimen_mismatch':
            # Serum test marked as urine specimen
            record['SampleType'] = 'Urine'
            # But test values are clearly serum-based
            record['Creatinine'] = round(np.random.uniform(0.8, 1.2), 2)  # Serum range
            
        elif alteration_type == 'extreme_values':
            # Biologically implausible values
            extreme_test = random.choice(['Hemoglobin', 'Creatinine', 'Sodium', 'Potassium'])
            if extreme_test == 'Hemoglobin':
                record['Hemoglobin'] = round(np.random.uniform(2, 4), 1)  # Extremely low
            elif extreme_test == 'Creatinine':
                record['Creatinine'] = round(np.random.uniform(8, 15), 1)  # Extremely high
            elif extreme_test == 'Sodium':
                record['Sodium'] = round(np.random.uniform(110, 120), 1)  # Dangerously low
            elif extreme_test == 'Potassium':
                record['Potassium'] = round(np.random.uniform(7, 9), 1)  # Dangerously high
        
        return record

    def generate_dataset(self, n_records: int = 1000, altered_ratio: float = 0.5) -> pd.DataFrame:
        """Generate complete synthetic dataset"""
        records = []
        n_altered = int(n_records * altered_ratio)
        n_unaltered = n_records - n_altered
        
        # Generate date range (last 2 years)
        start_date = datetime.now() - timedelta(days=730)
        end_date = datetime.now()
        
        # Track candidates to ensure blood type consistency
        candidate_records = {}
        
        # Generate unaltered records
        for i in range(n_unaltered):
            report_id = f"RPT_{str(i+1).zfill(7)}"
            candidate_id = f"C_{str(random.randint(1, n_records//3)).zfill(5)}"  # Some candidates have multiple reports
            
            # Random date
            days_diff = (end_date - start_date).days
            random_days = random.randint(0, days_diff)
            report_date = (start_date + timedelta(days=random_days)).strftime('%Y-%m-%d')
            
            record = self.generate_normal_record(report_id, candidate_id, report_date)
            records.append(record)
            
            # Track candidate blood type
            if candidate_id not in candidate_records:
                candidate_records[candidate_id] = {
                    'ABO_BloodGrp': record['ABO_BloodGrp'],
                    'Rh_Factor': record['Rh_Factor']
                }
        
        # Generate altered records
        for i in range(n_altered):
            report_id = f"RPT_{str(n_unaltered + i + 1).zfill(7)}"
            candidate_id = f"C_{str(random.randint(1, n_records//3)).zfill(5)}"
            
            # Random date
            days_diff = (end_date - start_date).days
            random_days = random.randint(0, days_diff)
            report_date = (start_date + timedelta(days=random_days)).strftime('%Y-%m-%d')
            
            record = self.generate_altered_record(report_id, candidate_id, report_date)
            records.append(record)
        
        # Shuffle records
        random.shuffle(records)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Reorder columns to match specification
        column_order = [
            'report_id', 'candidate_id', 'report_date',
            'Hemoglobin', 'Hematocrit', 'WBC', 'Neutrophils', 'Lymphocytes', 'Platelets',
            'ALT', 'AST', 'Bilirubin', 'HBsAg', 'HBV_DNA', 'HAV_IgM', 'RPR', 'Treponemal',
            'TotalProtein', 'Albumin', 'HbA1c', 'FastingGlucose', 'Creatinine', 'BUN',
            'Sodium', 'Potassium', 'ALP', 'Cholesterol', 'LDL', 'HDL', 'Triglycerides', 'VLDL',
            'Urine_RBC', 'Urine_Blood', 'Urine_Protein', 'ABO_BloodGrp', 'Rh_Factor',
            'Parasites', 'Eosinophils', 'RandomGlucose', 'SampleType', 'Target_Label'
        ]
        
        df = df[column_order]
        
        return df

# Generate the synthetic dataset (100 records for testing)
generator = MedicalDataGenerator()
synthetic_data = generator.generate_dataset(n_records=100, altered_ratio=0.5)

# Display basic statistics
print("Dataset Overview:")
print(f"Total records: {len(synthetic_data)}")
print(f"Altered records: {len(synthetic_data[synthetic_data['Target_Label'] == 'Altered'])}")
print(f"Unaltered records: {len(synthetic_data[synthetic_data['Target_Label'] == 'Unaltered'])}")
print(f"\nColumns: {synthetic_data.columns.tolist()}")
print(f"\nFirst 5 records:")
print(synthetic_data.head())

# Save to CSV
synthetic_data.to_csv('synthetic_medical_test_dataset.csv', index=False)
print(f"\nTest dataset saved as 'synthetic_medical_test_dataset.csv'")

# Display some example altered vs unaltered records
print("\n" + "="*80)
print("EXAMPLE UNALTERED RECORD:")
unaltered_example = synthetic_data[synthetic_data['Target_Label'] == 'Unaltered'].iloc[0]
for col in ['report_id', 'HbA1c', 'FastingGlucose', 'ALT', 'AST', 'Bilirubin', 'RPR', 'Treponemal']:
    print(f"{col}: {unaltered_example[col]}")

print("\n" + "="*80)
print("EXAMPLE ALTERED RECORD:")
altered_example = synthetic_data[synthetic_data['Target_Label'] == 'Altered'].iloc[0]
for col in ['report_id', 'HbA1c', 'FastingGlucose', 'ALT', 'AST', 'Bilirubin', 'RPR', 'Treponemal']:
    print(f"{col}: {altered_example[col]}")

print("\n" + "="*80)
print("VALUE RANGES REFERENCE:")
print("Normal HbA1c: 4.0-5.6% | Normal Fasting Glucose: 70-99 mg/dL")
print("Normal ALT: 7-56 U/L | Normal AST: 10-40 U/L")
print("RPR positive should typically have Treponemal positive for consistency")