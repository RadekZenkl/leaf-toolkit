# leaf-toolkit

This package is for testing and evaluating the Leaf Evaluation and Analysis Framework (LEAF). Please note that especially the GPU version is not runtime optimized. The goal is rather to provide an easy to use tool.
Pull requests for better runtime are welcome.

## Installation 
```
git clone https://github.com/RadekZenkl/leaf-toolkit.git
cd leaf-toolkit
pip install -e .
```

If you are going to utilize QR code detection please install pyzbar on your system: `sudo apt-get install libzbar0`  
More information about pyzbar [here](https://pypi.org/project/pyzbar/)


## Minimal Working Example

### Generating A4 Sheets with Sample QR Codes
The [bash script from Stewart et al (2016)](https://apsjournals.apsnet.org/doi/suppl/10.1094/PHYTO-01-16-0018-R) is a part of this project. In additional, it has been translated to python. It expects a `.txt` file in the following format:
```
sample1
sample2
...
```
The script can be run as follows:
```
from leaf.generate_sheets import generate_sheets

generate_sheets()
```

### Preparing Data for inference

When you want to apply the leaf-toolkit onto a scanned A4 sheets generated by the `generate_sheets` or with the provided [bash script from Stewart et al (2016)](https://apsjournals.apsnet.org/doi/suppl/10.1094/PHYTO-01-16-0018-R) the images need to pre preprocessed before: 

```
from leaf import prepare_folder

src_path = '<path to the directory with scanned images>'
export_path = '<path where to save the cropped samples>'
error_logs_path = '<name of the error log >'

prepare_folder(src_path, export_path, error_logs_path, manual_correction=True)
```

for example:

```
from leaf.data_prep import prepare_folder

src_path = 'scanned_images'
export_path = 'export'
error_logs_path = 'errors.txt'

prepare_folder(src_path, export_path, error_logs_path, manual_correction=True)
```

This code will find QR codes in the image, crop out individual leaves (1024x8196px) and save them in the export folder under the name encoded in the individual QR codes. If at least one QR code cannot be read (8 QR codes are expected) the image will be logged and later opened for manual inspection/correction. 

As long as the argument `manual_correction=True` the images with failed QR code detection will be showed in a pop-up window where the image crops can be done and the filename can be entered manually.

Already detected QR codes are denoted by blue bounding boxes. Please review which QR codes have been left out. Then left-click at a desired location to place the area for cropping which will be previewed with a red bounding box. Feel free to re-adjust by clicking at another location. 
Afterwards type in the name of the sample and click on `confirm`. Now the cropped image has been saved and the bounding box has turned green. If there are additional undetected QR codes, repeat the process. Once done click on `done` and next image will be shown.

### Predicting with leaf-toolkit Model 

Once your data is ready for prediction you can apply the `leafnet` to either individual images or folders:

```
from leaf.leafnet import Leafnet

net = Leafnet()
net.predict('<image path or folder path>')
```

You can set the `debug=True` to preview all the results without saving them. The output will be saved in `export` folder unless specified otherwise. The code saves the segmentation masks in `export/predictions` and visualized predictions in `export/visualization`. The predictions maks save the classes `background`, `leaf`, `lesion`, `pycndia`, `rust pustule` with the integer values of `0,1,2,3,4` respectively. Thus, the predictions appear pitch black to the naked eye. 

### Evaluating Diseases

If you want to evaulate the semented images for disease metrics you can use the following code:

```
from leaf.metrics import Evaluator

evaluator = Evaluator('results.csv')
evaluator.predict('export/predictions')
```

This will iterate through the images in the target folder and write the results into a `.csv` file. Currently supported metrics are:
- PLACL (Partial Leaf Area Covered in Lesions)
- Number of pycnidia
- Leaf Area in 1e6 pixels

## Issues

when compiling the `.tex` file from `generate_sheets.py` does not work, you can easily compile it in e.g. overleaf. Typically it has to do with the installation of `pdflatex` on your computer.

## Citation

@software{Leaf-Toolkit,
author = {Zenkl, Radek and Anderegg, Jonas and McDonald, Bruce},
license = {GPLv3},
month = sep,
title = {{leaf-toolkit}},
url = {https://github.com/RadekZenkl/leaf-toolkit},
version = {0.1.0},
year = {2023}
}
