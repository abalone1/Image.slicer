# Image.slicer
Slice images/labels with overlapping and in different scales

### Feature
- Preprocess of high resolution imagery and label data for use in Semantic Segmentation tasks (Deep Learning)
- Increases amount of trainingdata by generating different scales and overlappings for images and labels
- Multi stage interpolation 
- Easy to use and organize

# Example
```python

from exp.nb_Image import *

inp_d = Path("/Users/Abalone/Desktop/Test_vahingen/Vaihingen")
resize = 448 # Output size for the sliced Image
slice_range = [500, 900] # List of slice sizes in pixel
overlap = 7 # Ratio of overlapping (here 1:7)
dir_name = f'test4_{resize}_{overlap}_{slice_range[0]}-{slice_range[1]}' 
#Suggested directory name to save the outputs in

# Dictionary to convert rgb masks to greyscale
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

# Generate quantiles from slice_range
slice_l = ImageSlicer.quantile_from_slice_range(slice_range)

# Showing slices in Notebook
ImageSlicer.show_slices(slice_l, inp_d)
```
<img src="https://github.com/abalone1/Image_slicer_remote/blob/master/pic/show_slices.png" width="400">

```python
# For Image data
ImageSlicer.slice_images(inp_d= inp_d, dir_name = dir_name, slice_l = slice_l ,
                         resize=resize, overlap = overlap)        
```
<img src="https://github.com/abalone1/Image_slicer_remote/blob/master/pic/Image_slicer.png" width="600">

```python

# For label data
inp_d_label = Path("/Users/Abalone/Desktop/Test_vahingen/vaihingen_mask")
ImageSlicer.slice_masks(inp_d= inp_d2, dir_name = dir_name, slice_l = slice_l ,
                       resize= resize , overlap = overlap , palette = palette)
                       
```
<img src="https://github.com/abalone1/Image_slicer_remote/blob/master/pic/Mask_slicer.png" width="700">

#### Parameter

**inp_d** - Path("PATH TO IMAGE DIRECTORY") <br />
**dir_name** - Name of the directory where the files should be stored <br />
**resize** - In pixel: What should be the dimension of the sliced Images? <br />
**slice_range** - # In pixel: List of slice sizes <br />
**overlap** - How much Overlap (ratio) 1:? on each side? <br />

<img src="https://github.com/abalone1/Image_slicer_remote/blob/master/pic/Overlap.png" width="500">

#### optional
**out_path** - Path to output directory <br />
**palette** -  Dictionary to convert rgb masks to greyscale <br />
**slice_l** -  *ImageSlicer.quantile_from_slice_range* generates quantile from **'slice_range'** <br />

## Why this tool?

-  Training of high-resolution images is costly. Slicing can reduce costs. Padding and overlap can help to decrease the loss of information at the borders of the sliced tiles.
-  The use of different scales increases training data and can probably make the model more robust in production

This is a alterd version from [AnmolChachra/Image_slice]( https://github.com/AnmolChachra/Image-Slicer)
