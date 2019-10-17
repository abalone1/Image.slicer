# Image.slicer
Slice images/labels with overlapping and in different scales
Predict (with fastai) and merge the sliced images to full size

### Feature
- Preprocess of high resolution imagery and label data for use in Semantic Segmentation tasks (Deep Learning)
- Increases amount of trainingdata by generating different scales and overlappings for images and labels
- Multi stage interpolation (Nearest Neighbor + Bicubic combined) for image data 
- Nearest Neighbor interpolation for label data
- More than half empty slices will be ignored / It´s possilbe to slice a dismembered Mosaik!
- Add Padding (to the right and bottom) to your high resolution images
- Do Fastai predictions and merge the images to full size

# Example
```python

from notebook.exp.Image_Slicer import *

inp_d = Path("/Users/Abalone/Desktop/Vaihingen") 
# Directory where the Raw Images are stored
resize = 448 # Output size for the sliced Images
slice_range = [500, 900] # List of slice sizes in pixel
overlap = 7 # Ratio of overlapping (here 1:7 see image below)
dir_name = f'test4_{resize}_{overlap}_{slice_range[0]}-{slice_range[1]}' 
#dir_name = test4_448_7_500-900
#Directory name to save the outputs in 

# Dictionary to convert rgb masks to greyscale
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

# Generate quantiles from slice_range (0% , 25% , 50% , 75% , 100%)
slice_l = ImageSlicer.quantile_from_slice_range(slice_range)
slice_l = [500 , 600 , 700, 800 , 900]

# Showing slices in Notebook
ImageSlicer.show_slices(slice_l, inp_d)
```
<img src="https://github.com/abalone1/Image.slicer/blob/master/pic/show_slices.png" width="400">

```python
# For Image data
ImageSlicer.slice_images(inp_d= inp_d, dir_name = dir_name, slice_l = slice_l ,
                         resize=resize, overlap = overlap)        
```
<img src="https://github.com/abalone1/Image.slicer/blob/master/pic/Image_slicer.png" width="600">

```python

# For label data
inp_d_label = Path("/Users/Abalone//Test_vahingen/vaihingen_mask")
ImageSlicer.slice_masks(inp_d= inp_d2, dir_name = dir_name, slice_l = slice_l ,
                       resize= resize , overlap = overlap , palette = palette)
                       
```
<img src="https://github.com/abalone1/Image.slicer/blob/master/pic/Mask_slicer.png" width="700">

#### Parameter

**inp_d** - Path("PATH TO IMAGE DIRECTORY") <br />
**dir_name** - Name of the directory where the images or labels are stored <br />
**resize** - In pixel: What should be the dimension of the sliced Images? <br />
**slice_range** - # In pixel: List of slice sizes <br />
**overlap** - How much Overlap (ratio) 1:? on each side? <br />
<p align="center">
<img src="https://github.com/abalone1/Image.slicer/blob/master/pic/Overlap2.png" width="700">
</p>

#### optional
**out_path** - Path to output directory <br />
**palette** -  Dictionary to convert rgb masks to greyscale <br />
**slice_l** -  *ImageSlicer.quantile_from_slice_range* generates quantile from **'slice_range'** <br />

## Predict and merge

```python
from fastai.vision import *
from PIL import Image

def acc_isprs(input, target):
    target = target.squeeze(1)
    mask = target != void_code 
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

metrics = acc_isprs
f_score1 = partial(fbeta , beta=1)

path_model  = Path('D:/david/va_po/im_train/va_5-9_po__9-13_448')

inp_d = Path("D:/david/Test_data/test7/out_padded/po_448_0_1100-1100")

rows, columns = 7,7

ImageSlicer.predict_from_fastai_model(path_fastai_model= path_model,inp_d= inp_d, dir_name=dir_name, palette= palette , rows= rows, columns= columns )
```
This is a alterd version from [AnmolChachra/Image_slice]( https://github.com/AnmolChachra/Image-Slicer)
