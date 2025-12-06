import torch
import cv2
import os
import logging
import numpy as np
from archs.iart_arch import IART
from basicsr.utils import tensor2img, get_root_logger, get_time_str
from basicsr.metrics import psnr_ssim
from torchvision import transforms
from PIL import Image

def load_images_to_tensor(folder_path):
    """
    Returns:
        tuple: (Tensor of shape [N, C, H, W], List of filenames)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    images_tensors = []
    valid_filenames = [] # Store the actual filenames here
    
    # Sort to ensure frames are in correct order
    file_list = sorted(os.listdir(folder_path))

    for filename in file_list:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # PIL opens as RGB
            image = Image.open(os.path.join(folder_path, filename))
            image_tensor = transform(image)
            images_tensors.append(image_tensor)
            valid_filenames.append(filename)

    if not images_tensors:
        return None, None

    images_tensor = torch.stack(images_tensors)
    return images_tensor, valid_filenames

@torch.no_grad()
def main():
    device = torch.device('cuda')
    
    # --- CONFIGURATION ---
    input_root = 'demo/Vid4_BI'
    gt_root = 'demo/Vid4'
    
    output_root = 'results/test_data/Vid4_frames_upscaled'
    # output_root = 'results/test_data/Vid4_upscaled_dynamic_improved'
    
    model_path = 'experiments/IART_REDS_N5_60K/models/net_g_60000.pth'
    # model_path = 'experiments/IART_REDS_N5_60K_Improved_Dynamic/models/net_g_60000.pth'
    
    CHUNK_SIZE = 10
    crop_border = 0
    test_y_channel = False

    # --- LOGGER ---
    os.makedirs(output_root, exist_ok=True)
    log_file = os.path.join(output_root, f'psnr_ssim_test_{get_time_str()}.log')
    logger = get_root_logger(logger_name='recurrent', log_level=logging.INFO, log_file=log_file)
    logger.info(f'Data: {input_root}')
    logger.info(f'GT: {gt_root}')

    # --- MODEL ---
    model = IART(mid_channels=64, embed_dim=120, depths=[6, 6, 6], num_heads=[6, 6, 6],
                 window_size=[3, 8, 8], num_frames=3, cpu_cache_length=100, is_low_res_input=True,
                 spynet_path='experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth')
    
    checkpoint = torch.load(model_path)
    if 'params' in checkpoint: model.load_state_dict(checkpoint['params'], strict=False)
    else: model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model = model.to(device)

    avg_psnr_l = []
    avg_ssim_l = []
    
    # Get subfolders (calendar, city, foliage, walk)
    subfolders = sorted([f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))])

    for name in subfolders:
        logger.info(f"Processing Clip: {name}")
        
        video_path = os.path.join(input_root, name)
        gt_video_path = os.path.join(gt_root, name)
        save_path =  os.path.join(output_root, name)
        os.makedirs(save_path, exist_ok=True)
        
        # Check if matching GT folder exists
        if not os.path.exists(gt_video_path):
            logger.warning(f"GT folder not found for {name}. Skipping metrics for this clip.")
            # You might want to continue just for inference, but for PSNR test we skip
            continue

        # Load images AND filenames
        imgs_lq_full, filename_list = load_images_to_tensor(video_path)
        
        if imgs_lq_full is None: continue

        total_frames = imgs_lq_full.shape[0]
        clip_psnr = 0.0
        clip_ssim = 0.0
        processed_count = 0

        for i in range(0, total_frames, CHUNK_SIZE):
            sub_imgs = imgs_lq_full[i : i + CHUNK_SIZE]
            if sub_imgs.shape[0] == 0: continue

            sub_imgs_gpu = sub_imgs.unsqueeze(0).to(device)

            try:
                sub_outputs = model(sub_imgs_gpu).squeeze(0)
            except torch.OutOfMemoryError:
                logger.error("❌ OOM Error! Reduce CHUNK_SIZE.")
                torch.cuda.empty_cache()
                return

            for frame_idx in range(sub_outputs.shape[0]):
                # 1. Identify the correct filename
                # i is the chunk start index, frame_idx is index within chunk
                absolute_index = i + frame_idx
                save_filename = filename_list[absolute_index] # Use original filename (e.g., 00000.png)

                # 2. Convert directly to BGR (0-255)
                output_img = tensor2img(sub_outputs[frame_idx], rgb2bgr=True, min_max=(0, 1))
                
                # 3. Save using OpenCV (Expects BGR)
                cv2.imwrite(os.path.join(save_path, save_filename), output_img)
                
                # 4. Load GT using UNCHANGED
                gt_img_path = os.path.join(gt_video_path, save_filename)
                
                if os.path.exists(gt_img_path):
                    img_gt = cv2.imread(gt_img_path, cv2.IMREAD_UNCHANGED)
                    
                    # --- SAFETY CHECK FOR 16-BIT IMAGES ---
                    if img_gt.dtype == np.uint16:
                        # print(f"Converting 16-bit GT to 8-bit: {save_filename}") # Optional print
                        img_gt = (img_gt / 256).astype('uint8')
                    # --------------------------------------

                    # Resize if needed (safety)
                    if img_gt.shape[:2] != output_img.shape[:2]:
                        print(f"Resize warning: GT {img_gt.shape} vs Out {output_img.shape}")
                        h, w = output_img.shape[:2]
                        img_gt = cv2.resize(img_gt, (w, h))

                    # Calculate Metrics
                    crt_psnr = psnr_ssim.calculate_psnr(output_img, img_gt, crop_border=crop_border, test_y_channel=test_y_channel)
                    crt_ssim = psnr_ssim.calculate_ssim(output_img, img_gt, crop_border=crop_border, test_y_channel=test_y_channel)
                    
                    clip_psnr += crt_psnr
                    clip_ssim += crt_ssim
                    processed_count += 1
                    
                    logger.info(f'{name}--{save_filename} - PSNR: {crt_psnr:.6f} dB. SSIM: {crt_ssim:.6f}')

                else:
                    logger.warning(f"GT img file {gt_img_path} not found!")

            del sub_imgs_gpu
            del sub_outputs
            torch.cuda.empty_cache()

        if processed_count > 0:
            avg_psnr = clip_psnr / processed_count
            avg_ssim = clip_ssim / processed_count
            avg_psnr_l.append(avg_psnr)
            avg_ssim_l.append(avg_ssim)
            logger.info(f'Folder {name} - Average PSNR: {avg_psnr:.6f} dB. Average SSIM: {avg_ssim:.6f}.')

    if avg_psnr_l:
        logger.info(f'Average PSNR: {sum(avg_psnr_l) / len(avg_psnr_l):.6f} dB')
        logger.info(f'Average SSIM: {sum(avg_ssim_l) / len(avg_ssim_l):.6f}')
    
    print("Done.")

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
