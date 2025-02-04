# EGSM Enhancing Green Screen Matting

## Important Notice:
##The Adobe dataset has legal and responsibility restrictions and must not be shared arbitrarily!!!

Environment:  
Python 3.6  
CUDA 10.0  
PyTorch 1.7.1  
TensorFlow-GPU 1.14.0  

## 1. Dataset Processing - train_preprocessing
### (1) Synthetic Dataset: data_adobe

Dataset Composition:
```bash
python compose.py --fg_path fg_train --mask_path mask_train --bg_path bg_train --out_path merged_train --out_csv Adobe_train_data.csv --workers 8
```

Background blurring: 
```bash
python bg_fuzzy_process.py
```
Foreground-background denoising: 
```bash
python gaussian_filter_denoising.py
```
Color correction:
```bash
python color_correction.py
```

Training:
```bash
python train_adobe_*.py -n train_adobe_* -bs 4 -res 256
```

(2) Real Dataset: real_data
Process the video using AE software to generate:
Foreground images (*_fg.jpg)
Mask images (*_maskDL.jpg)
Original images (*_img.png)

Run the following script to process the dataset:
```bash
python ae_real.py
```

Note: When training, merge the generated *.csv files.


## 2. Network Architecture
Group Normalization: train_code_norm
ASPP Module: train_code_aspp
DPN Module: train_code_dpn
Perceptual Loss: train_code_VGG
   
All training processes use the respective train_adobe_*.py scripts from each folder.

Overall Network Structure: mine_train_final
    Training: python mine_train.py

## 3. Testing - test_code
Run segmentation test:
```bash
python test_segmentation_deeplab.py -i test_code/input
```
Run background matting test:
```bash
python test_background-matting_image_1.py  -m mine_train_0120 -o test/output -i test/input -tb test/background/0001.png -b test/bg.png
```

Trained Models Location:
green_screen_matting\test_code\Models
