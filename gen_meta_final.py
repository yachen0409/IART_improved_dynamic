import os
import cv2
import glob

def generate_meta_info(name, root_path):
    txt_file = f'datasets/meta_info/meta_info_{name}.txt'
    
    # Get all subfolders (000, 001, etc.)
    clip_list = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
    
    print(f"Scanning {root_path}...")
    print(f"Found {len(clip_list)} clips. Generating {txt_file}...")
    
    with open(txt_file, 'w') as f:
        for clip in clip_list:
            clip_path = os.path.join(root_path, clip)
            
            # Find all images (png, jpg, jpeg)
            frame_list = sorted(glob.glob(os.path.join(clip_path, '*.[jp][np]g')))
            
            if len(frame_list) == 0:
                print(f"⚠️ Warning: Empty folder found: {clip}")
                continue
                
            num_frames = len(frame_list)
            
            # Read the first image to get dimensions (Height, Width, Channels)
            img = cv2.imread(frame_list[0])
            if img is None:
                print(f"⚠️ Error reading image: {frame_list[0]}")
                continue
                
            h, w, c = img.shape
            
            # Write format: FolderName NumFrames (H,W,C)
            # Example: 000 100 (720,1280,3)
            info_line = f'{clip} {num_frames} ({h},{w},{c})'
            f.write(f'{info_line}\n')
            
    print(f"✅ Success! Meta info saved to {txt_file}")

# Run for TRAIN data only
if __name__ == "__main__":
    # Ensure this path matches your folder structure exactly
    generate_meta_info('REDS_GT', 'datasets/REDS/train_sharp')
