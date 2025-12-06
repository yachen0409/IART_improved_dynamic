import re
import csv
import os

def extract_validation_data(log_file_path):
    """Extract validation data from a single log file"""
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

def combine_validation_data(file_list):
    """Combine validation data from multiple log files"""
    combined_data = {}
    
    for file_path in file_list:
        data = extract_validation_data(file_path)
        combined_data.update(data)
    
    return combined_data

def main():
    # Define file paths
    baseline_files = [
        "IART_REDS_N5_15K/train_IART_REDS_N5_15K_20251121_170730.log",
        "IART_REDS_N5_15K/train_IART_REDS_N5_15K_20251121_202949.log"
    ]
    
    improved_files = [
        "IART_REDS_N5_15K_Improved/train_IART_REDS_N5_15K_Improved_20251122_111025.log",
        "IART_REDS_N5_15K_Improved/train_IART_REDS_N5_15K_Improved_20251122_152840.log"
    ]
    
    improved_dynamic_files = [
        "IART_REDS_N5_15K_Improved_Dynamic/train_IART_REDS_N5_15K_Improved_Dynamic_20251123_003633.log"
    ]
    
    # Extract validation data from each group
    print("Extracting validation data...")
    baseline_data = combine_validation_data(baseline_files)
    improved_data = combine_validation_data(improved_files)
    improved_dynamic_data = combine_validation_data(improved_dynamic_files)
    
    print(f"Baseline data points: {list(baseline_data.keys())}")
    print(f"Improved data points: {list(improved_data.keys())}")
    print(f"Improved Dynamic data points: {list(improved_dynamic_data.keys())}")
    
    # Define target iterations
    target_iterations = [5000, 7500, 10000]
    
    # Prepare CSV data
    csv_data = []
    
    for iteration in target_iterations:
        row = [iteration]
        
        # Baseline data
        if iteration in baseline_data:
            row.extend([baseline_data[iteration]['psnr'], baseline_data[iteration]['ssim']])
        else:
            row.extend(['', ''])  # Empty if no data
        
        # Improved data
        if iteration in improved_data:
            row.extend([improved_data[iteration]['psnr'], improved_data[iteration]['ssim']])
        else:
            row.extend(['', ''])  # Empty if no data
        
        # Improved Dynamic data
        if iteration in improved_dynamic_data:
            row.extend([improved_dynamic_data[iteration]['psnr'], improved_dynamic_data[iteration]['ssim']])
        else:
            row.extend(['', ''])  # Empty if no data
        
        csv_data.append(row)
    
    # Write to CSV
    output_file = "preliminary_validation_data_10k.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['iteration', 'psnr_baseline', 'ssim_baseline', 'psnr_improved', 'ssim_improved', 'psnr_improved_dynamic', 'ssim_improved_dynamic']
        writer.writerow(header)
        
        # Write data
        writer.writerows(csv_data)
    
    print(f"\nCSV file '{output_file}' has been created successfully!")
    
    # Display the data for verification
    print("\nExtracted data:")
    print("=" * 80)
    for i, row in enumerate(csv_data):
        iteration = row[0]
        print(f"Iteration {iteration}:")
        if row[1] and row[2]:
            print(f"  Baseline: PSNR={row[1]}, SSIM={row[2]}")
        if row[3] and row[4]:
            print(f"  Improved: PSNR={row[3]}, SSIM={row[4]}")
        if row[5] and row[6]:
            print(f"  Improved Dynamic: PSNR={row[5]}, SSIM={row[6]}")
        print()

if __name__ == "__main__":
    main()
