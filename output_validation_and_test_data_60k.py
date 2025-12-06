import re
import csv
import os

def extract_validation_data(log_file_path):
    """Extract validation data from training log files"""
    validation_data = {}
    
    if not os.path.exists(log_file_path):
        print(f"Warning: File {log_file_path} not found")
        return validation_data
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Pattern to match validation blocks
            validation_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO: Validation REDS4\s+# psnr: ([\d.]+).*?# ssim: ([\d.]+).*?Best: [\d.]+ @ (\d+) iter'
            
            matches = re.findall(validation_pattern, content, re.DOTALL)
            
            for match in matches:
                timestamp, psnr, ssim, iteration = match
                iteration = int(iteration)
                validation_data[iteration] = {
                    'psnr': float(psnr),
                    'ssim': float(ssim),
                    'timestamp': timestamp
                }
                
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
    
    return validation_data

def extract_test_data(log_file_path):
    """Extract test data from PSNR/SSIM test log files"""
    test_data = {}
    
    if not os.path.exists(log_file_path):
        print(f"Warning: File {log_file_path} not found")
        return test_data
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Pattern to match folder average results
            folder_pattern = r'Folder (\w+) - Average PSNR: ([\d.]+) dB\. Average SSIM: ([\d.]+)\.'
            
            matches = re.findall(folder_pattern, content)
            
            for match in matches:
                video_name, psnr, ssim = match
                test_data[video_name] = {
                    'psnr': float(psnr),
                    'ssim': float(ssim)
                }
                
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
    
    return test_data

def combine_validation_data(file_list):
    """Combine validation data from multiple log files"""
    combined_data = {}
    
    for file_path in file_list:
        data = extract_validation_data(file_path)
        combined_data.update(data)
    
    return combined_data

def combine_test_data(file_list):
    """Combine test data from multiple log files"""
    combined_data = {}
    
    for file_path in file_list:
        data = extract_test_data(file_path)
        combined_data.update(data)
    
    return combined_data

def main():
    # Define file paths
    baseline_files = [
        "experiments/IART_REDS_N5_60K/train_IART_REDS_N5_60K_20251123_123733.log",
        "experiments/IART_REDS_N5_60K/train_IART_REDS_N5_60K_20251125_155521.log"
    ]
    
    improved_dynamic_files = [
        "experiments/IART_REDS_N5_60K_Improved_Dynamic/train_IART_REDS_N5_60K_Improved_Dynamic_20251125_230122.log"
    ]
    
    test_baseline_files = [
        "results/test_data/Vid4_frames_upscaled/psnr_ssim_test_20251203_224952.log"
    ]
    
    test_improved_dynamic_files = [
        "results/test_data/Vid4_upscaled_dynamic_improved/psnr_ssim_test_20251203_224826.log"
    ]
    
    # ========== PART 1: Extract validation data ==========
    print("Extracting validation data...")
    baseline_validation = combine_validation_data(baseline_files)
    improved_dynamic_validation = combine_validation_data(improved_dynamic_files)
    
    print(f"Baseline validation data points: {list(baseline_validation.keys())}")
    print(f"Improved Dynamic validation data points: {list(improved_dynamic_validation.keys())}")
    
    # Get validation data every 10000 iterations
    validation_iterations = []
    max_iter = max(list(baseline_validation.keys()) + list(improved_dynamic_validation.keys())) if (baseline_validation or improved_dynamic_validation) else 0
    
    for i in range(10000, max_iter + 1, 10000):
        validation_iterations.append(i)
    
    # Prepare validation CSV data
    validation_csv_data = []
    
    for iteration in validation_iterations:
        row = [iteration]
        
        # Baseline validation data
        if iteration in baseline_validation:
            row.extend([baseline_validation[iteration]['psnr'], baseline_validation[iteration]['ssim']])
        else:
            row.extend(['', ''])  # Empty if no data
        
        # Improved Dynamic validation data
        if iteration in improved_dynamic_validation:
            row.extend([improved_dynamic_validation[iteration]['psnr'], improved_dynamic_validation[iteration]['ssim']])
        else:
            row.extend(['', ''])  # Empty if no data
        
        validation_csv_data.append(row)
    
    # Write validation CSV
    validation_output_file = "experiments/validation_data_60k.csv"
    os.makedirs(os.path.dirname(validation_output_file), exist_ok=True)
    
    with open(validation_output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['iteration', 'psnr_baseline', 'ssim_baseline', 'psnr_improved_dynamic', 'ssim_improved_dynamic']
        writer.writerow(header)
        
        # Write data
        writer.writerows(validation_csv_data)
    
    print(f"\nValidation CSV file '{validation_output_file}' has been created successfully!")
    
    # ========== PART 2: Extract test data ==========
    print("\nExtracting test data...")
    baseline_test = combine_test_data(test_baseline_files)
    improved_dynamic_test = combine_test_data(test_improved_dynamic_files)
    
    print(f"Baseline test videos: {list(baseline_test.keys())}")
    print(f"Improved Dynamic test videos: {list(improved_dynamic_test.keys())}")
    
    # Get all unique video names
    all_videos = set(list(baseline_test.keys()) + list(improved_dynamic_test.keys()))
    
    # Prepare test CSV data
    test_csv_data = []
    
    for video_name in sorted(all_videos):
        row = [video_name]
        
        # Baseline test data
        if video_name in baseline_test:
            row.extend([baseline_test[video_name]['psnr'], baseline_test[video_name]['ssim']])
        else:
            row.extend(['', ''])  # Empty if no data
        
        # Improved Dynamic test data
        if video_name in improved_dynamic_test:
            row.extend([improved_dynamic_test[video_name]['psnr'], improved_dynamic_test[video_name]['ssim']])
        else:
            row.extend(['', ''])  # Empty if no data
        
        test_csv_data.append(row)
    
    # Write test CSV
    test_output_file = "results/test_data/test_data_60k.csv"
    os.makedirs(os.path.dirname(test_output_file), exist_ok=True)
    
    with open(test_output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['video_name', 'psnr_baseline', 'ssim_baseline', 'psnr_improved_dynamic', 'ssim_improved_dynamic']
        writer.writerow(header)
        
        # Write data
        writer.writerows(test_csv_data)
    
    print(f"Test CSV file '{test_output_file}' has been created successfully!")
    
    # Display the data for verification
    print("\n" + "="*80)
    print("VALIDATION DATA SUMMARY:")
    print("="*80)
    for i, row in enumerate(validation_csv_data):
        iteration = row[0]
        print(f"Iteration {iteration}:")
        if row[1] and row[2]:
            print(f"  Baseline: PSNR={row[1]}, SSIM={row[2]}")
        if row[3] and row[4]:
            print(f"  Improved Dynamic: PSNR={row[3]}, SSIM={row[4]}")
        print()
    
    print("="*80)
    print("TEST DATA SUMMARY:")
    print("="*80)
    for i, row in enumerate(test_csv_data):
        video_name = row[0]
        print(f"Video: {video_name}")
        if row[1] and row[2]:
            print(f"  Baseline: PSNR={row[1]}, SSIM={row[2]}")
        if row[3] and row[4]:
            print(f"  Improved Dynamic: PSNR={row[3]}, SSIM={row[4]}")
        print()

if __name__ == "__main__":
    main()
