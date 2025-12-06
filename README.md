# CSE881 Project -- Improve existing super-resolution, IART

- Please note that all the following commands assumed you are (1) using command prompt to give commands and (2) at the project folder root (IART_improved_dynamic).

## Installation

* We ran the code with **Python 3.9** and the packages list is in "requirements.txt". Install the packages by `pip install -r requirements.txt`.
  - Since they are lots of packages, we recommend you to create a virtual environment (venv) using Python 3.9 first, then install the packages. 
* The **Version of PyTorch + cuda** in project_requirements.txt is for NVIDIA RTX5090.
  - If you have other GPU, change the pytorch + cuda version that are compatible to your GPU.

## Datasets Path
- Training
  - REDS (270 videos)
    - The original video frames (ground truth) are in "datasets/REDS/train_sharp/{$video_number}"
    - The downsampled video frames (4x downgrade) are in "datasets/REDS/train_sharp_bicubic/{$video_number}"
  - The data is too large. Therefore, only 5 videos are in this repository
- Validation
  - REDS (30 videos)
    - The original video frames (ground truth) are in "datasets/REDS/val_sharp/{$video_number}"
    - The downsampled video frames (4x downgrade) are in "datasets/REDS/val_sharp_bicubic/{$video_number}"
  - The data is too large. Therefore, only 1 video is in this repository
- Testing
  - Vid4 (4 videos)
    - The original video frames (ground truth) are in "demo/Vid4/{$video_name}"
    - The downsampled video frames (4x downgrade) are in "demo/Vid4_BI/{$video_name}"

## Training (include validation) 
- Train baseline model
  - 10k iterations
    - `python recurrent_mix_precision_train.py -opt options/train/train_baseline_10k.yml`
    - The output models, training states, and the training logs will be at "experiments/IART_REDS_N5_10K/"
  - 60k iterations
    - `python recurrent_mix_precision_train.py -opt options/train/train_baseline_60k.yml`
    - The output models, training states, and the training logs will be at "experiments/IART_REDS_N5_60K/"
- Train fixed threshold model
  - 10k iterations
    - `python recurrent_mix_precision_train_improved.py -opt options/train/train_improved_10k.yml`
    - The output models, training states, and the training logs will be at "experiments/IART_REDS_N5_10K_Improved/"
- Train dynamic weighting model
  - 10k iterations
    - `python recurrent_mix_precision_train_improved_dynamic.py -opt options/train/train_improved_dynamic_10k.yml`
    - The output models, training states, and the training logs will be at "experiments/IART_REDS_N5_10K_Improved_Dynamic/"
  - 60k iterations
    - `python recurrent_mix_precision_train_improved_dynamic.py -opt options/train/train_improved_dynamic_60k.yml`
    - The output models, training states, and the training logs will be at "experiments/IART_REDS_N5_60K_Improved_Dynamic/"

## Testing
- Test baseline model with 60k iterations
  - `python test_vid4_baseline_60k.py`
  - The output frames of 4 videos will be at "results/test_data/Vid4_frames_upscaled"
- Test dynamic weighting model with 60k iterations
  - `python test_vid4_dynamic_improved_60k.py`
  - The output frames of 4 videos will be at "results/test_data/Vid4_upscaled_dynamic_improved"


## Web Application
- Run the web locally
  - `cd webapp && streamlit run app.py`
  - The webpage should show up automatically. If not, the URL (ip:port) is output in the terminal. Please copy and paste the URL to a new private window to test the application.


## Results (Graphs/Figures) Reproduce
### Preliminary Validation
- Figures of loss curve over 10k iterations (both baseline vs fixed threshold and baseline vs dynamic weighting)
  - `cd experiments && python plot_loss_ori_fixed_10k.py && python plot_loss_ori_dynamic_10k.py`
  - The output figures are located at "experiments/"
- Tables of validation results on baseline model, fixed threshold model, and dynamic weighting model over 10k iterations (in csv format)
  - `cd experiments && python output_preliminary_validation_data_10k.py`
  - The output csv file is located at "experiments/preliminary_validation_data_10k.csv"

### Main Result (Experimental Results)
- Figure of loss curve over 60k iterations (baseline vs dynamic weighting)
  - `cd experiments && python plot_loss_ori_dynamic_60k.py`
- Tables of validation results and test results on baseline model and dynamic weighting model over 60k iterations (in csv format)
  - `python output_validation_and_test_data_60k.py`

