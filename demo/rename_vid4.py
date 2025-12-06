import os

def rename_vid4_frames():
    # --- CONFIGURATION ---
    target_root = 'Vid4'  # <--- Path to your Vid4 folder
    
    # True = Print what would happen (Safe)
    # False = Actually rename files (Destructive)
    DRY_RUN = False 
    # ---------------------

    if not os.path.exists(target_root):
        print(f"Error: Folder '{target_root}' not found.")
        return

    # Get subfolders (calendar, city, foliage, walk)
    subfolders = sorted([f for f in os.listdir(target_root) if os.path.isdir(os.path.join(target_root, f))])
    
    print(f"Found {len(subfolders)} subfolders: {subfolders}\n")

    for sub in subfolders:
        sub_path = os.path.join(target_root, sub)
        
        # Get all PNGs and sort them to ensure 001 comes before 002
        files = sorted([f for f in os.listdir(sub_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"📂 Processing: {sub} ({len(files)} frames)")

        for i, filename in enumerate(files):
            # i starts at 0. 
            # So the first file (e.g., 001.png) becomes 00000000.png
            new_name = f"{i:08d}.png"
            
            old_file_path = os.path.join(sub_path, filename)
            new_file_path = os.path.join(sub_path, new_name)

            # Skip if name is already correct
            if filename == new_name:
                continue

            if DRY_RUN:
                print(f"   [Would Rename] {filename}  -->  {new_name}")
            else:
                # Check if target exists to prevent overwriting (rare edge case)
                if os.path.exists(new_file_path):
                    print(f"   ❌ Error: Target {new_name} already exists. Skipping {filename}.")
                    continue
                
                os.rename(old_file_path, new_file_path)
        
        if not DRY_RUN:
            print("   ✅ Done.")
        print("-" * 20)

    if DRY_RUN:
        print("\n📢 SIMULATION COMPLETE. No files were changed.")
        print("Set DRY_RUN = False in the script to execute.")
    else:
        print("\n🎉 RENAMING COMPLETE.")

if __name__ == '__main__':
    rename_vid4_frames()
