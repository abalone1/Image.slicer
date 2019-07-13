
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: notebook/Image_slicer_scale.ipynb

%load_ext autoreload
%autoreload 2
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage import io
import PIL,os,mimetypes

Path.ls = lambda x: list(x.iterdir())

image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

class ImageSlicer():

    def __init__(self, source, slice_s, overlap):
        self.source = source
        self.slice_s = slice_s
        self.overlap = overlap

    def _get_files(p, fs, extensions=image_extensions):
        p = Path(p)
        res = [p/f for f in fs if not f.startswith('.')
               and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
        return res

    def _read_images(self):
        Images = []
        image_names = sorted(os.listdir(self.source))
        for im in image_names:
            image = plt.imread(os.path.join(dir_path,im))
            Images.append(image)
        return Images


    def _convolution(self, Image, strides, slice_s):
        start_x = 0
        start_y = 0
        n_rows = Image.shape[0]//strides[0] + 1
        minus = divmod(slice_s,strides[0])[0]
        n_columns = Image.shape[1]//strides[1] + 1
        small_images = []
        for i in range(n_rows-1):
            for j in range(n_columns-minus):
                new_start_x = start_x+i*strides[0]
                new_start_y= start_y+j*strides[1]
                small_images.append(Image[new_start_x:new_start_x+self.slice_s,new_start_y:new_start_y+self.slice_s,:])
        return small_images

    def _transform(self):

        if self.source:
            Image = plt.imread(self.source)
            Images = [Image]

        im_size = Images[0].shape
        num_images = len(Images)
        transformed_images = dict()
        Images = np.array(Images)

        stride_x = []
        stride_y = []

        if self.overlap == None:
            stride_x = self.slice_s
            stride_y = self.slice_s
        elif self.overlap == 0 or self.overlap == 1:
            stride_x = self.slice_s
            stride_y = self.slice_s
        elif self.overlap > 1:
            stride_x = self.slice_s/self.overlap*(self.overlap-1)
            stride_x = int(stride_x)
            stride_y = self.slice_s/self.overlap*(self.overlap-1)
            stride_y = int(stride_x)

        strides = [stride_x, stride_y]

        for i, Image in enumerate(Images):
            transformed_images[str(i)] = self._convolution(Image, strides, self.slice_s)

        return transformed_images

    def quantile_from_slice_range(slice_range):
        _slice_size_0 = int(slice_range[0])
        _slice_size_1 = int(np.percentile(slice_range, 25, axis=0))
        _slice_size_2 = int(np.percentile(slice_range, 50, axis=0))
        _slice_size_3 = int(np.percentile(slice_range, 75, axis=0))
        _slice_size_4 = int(slice_range[1])

        slice_l = [_slice_size_0,_slice_size_1 , _slice_size_2 ,  _slice_size_3 , _slice_size_4 ]
        return slice_l

    def show_slices(slice_l, inp_d , pos= None):

        if pos == None:
            pos = 0

        inp_d = Path(inp_d)

        sl = [o.name for o in os.scandir(inp_d)]
        sl = ImageSlicer._get_files(inp_d, sl)
        im = Image.open(sl[pos])

        cropped_l = []
        for i in slice_l:
            left, top, right, bottom = 0, 0, i, i
            c  = im.crop( ( left, top, right, bottom ) )
            cropped_l.append(c)

        Tot = number_of_subplots=len(slice_l)
        Tot = Tot +1
        Cols = 2
        Rows = Tot // Cols
        Rows += Tot % Cols
        Position = range(1,Tot)

        fig = plt.figure(figsize=(12,12))

        for j, k in enumerate(cropped_l):
            ax = fig.add_subplot(Rows,Cols,Position[j])
            ax.imshow(k,interpolation='lanczos' )
            ax.set_axis_off()
            ax.set_title(f'Slice size: {slice_l[j]}')
        plt.show()

    def slice_images(inp_d = None, dir_name = None, slice_l = None, resize= None, overlap = None, out_path= None, cb=None):

        inp_d = Path(inp_d)
        if out_path==None:
                    out_path = inp_d
        out_path = Path(inp_d)

        im_l = [o.name for o in os.scandir(inp_d)]
        im_l = ImageSlicer._get_files(inp_d, im_l)

        cb = PrintStatusCallback()

        for i in slice_l:
            last_slice = slice_l[-1]

            if cb: cb.before_calc(i)
            for p, im in enumerate(im_l):
                li = len(im_l)
                if cb: cb.status_calc(p , li)
                slicer = ImageSlicer(im, slice_s = i ,overlap=overlap)
                transformed_image = slicer._transform()
                ImageSlicer._save_images(transformed=transformed_image,resize=resize , out_path=out_path , dir_name=dir_name, _slice_size=i , overlap = overlap, im = im)
            if cb: cb.after_calc(i, last_slice, out_path/dir_name)

    def _save_images(transformed, out_path, dir_name, resize, _slice_size=int , overlap =int ,im = None):

        if type(resize) == tuple:
            resize_t = resize
        else:
            resize_t = ()
            resize_t = (resize , resize)

        (out_path/dir_name).mkdir(exist_ok=True)
        for key, val in transformed.items():
            shape = val[0].shape
            count = shape[0] * shape[0]
            count_h = count//2
            for k, j in enumerate(val):
                m = np.all(j == np.array((0,0,0)).reshape(1, 1, 3), axis=2)
                non_zero = np.count_nonzero(m)
                if non_zero <= count_h:
                    img = Image.fromarray(j, 'RGB')
                    diff = _slice_size - resize_t[0]
                    diff_h = diff//2 + resize_t[0]
                    img.resize((diff_h, diff_h),resample=Image.BICUBIC).resize(resize_t, resample= PIL.Image.NEAREST).save(out_path/dir_name/f'{im.stem}_{k}_{overlap}_{_slice_size}.png')


    def slice_masks(inp_d = None, dir_name = None ,slice_l = None,resize= None, overlap = None, out_path=None, cb=None, palette = None):

        inp_d = Path(inp_d)
        if out_path==None:
                    out_path = inp_d
        out_path = Path(inp_d)

        mask_l = [o.name for o in os.scandir(inp_d)]
        mask_l = ImageSlicer._get_files(inp_d, mask_l)

        cb = PrintStatusCallback()

        for i in slice_l:
            last_slice = slice_l[-1]

            if cb: cb.before_calc(i)
            for p, im in enumerate(mask_l):
                li = len(mask_l)
                if cb: cb.status_calc(p , li)
                slicer = ImageSlicer(im, slice_s = i ,overlap=overlap)
                transformed_image = slicer._transform()
                ImageSlicer._save_masks(transformed=transformed_image,resize=resize , out_path=out_path , dir_name=dir_name, _slice_size=i , overlap = overlap, im = im)
            if cb: cb.after_calc(i, last_slice, out_path/dir_name )

        if palette != None:
                    print("----")
                    print("Convert colour to grey")

                    mask_grey_l = [o.name for o in os.scandir(inp_d/dir_name)]
                    mask_grey_l = ImageSlicer._get_files(inp_d/dir_name, mask_grey_l)

                    for mask in mask_grey_l:
                        mask_r = io.imread(mask)
                        mask_c = ImageSlicer._color_to_grey(mask_rgb = mask_r , palette = palette, out_path =out_path/dir_name/dir_name,dir_name = dir_name , path=mask)

                    print(cb.after_calc(i,last_slice ,out_path/dir_name/dir_name ))


    def _save_masks(transformed, out_path, dir_name, resize, _slice_size=None , overlap =int,im = None):
        if type(resize) == tuple:
            resize_t = resize
        else:
            resize_t = ()
            resize_t = (resize , resize)

        (out_path/dir_name).mkdir(exist_ok=True)
        for key, val in transformed.items():
            shape = val[0].shape
            count = shape[0] * shape[0]
            count_h = count//2
            for k, j in enumerate(val):
                m = np.all(j == np.array((0,0,0)).reshape(1, 1, 3), axis=2)
                non_zero = np.count_nonzero(m)
                if non_zero <= count_h:
                    img = Image.fromarray(j, 'RGB')
                    img.resize(resize_t,resample=Image.NEAREST).save(out_path/dir_name/f'{im.stem}_{k}__{overlap}_{_slice_size}.png')

    def _color_to_grey(mask_rgb, palette=None, out_path=None, dir_name=None , path=None):
        """ RGB-color encoding to grayscale labels """
        if palette == None:
            raise Exception("Please insert palette")
        else:
            (out_path).mkdir(exist_ok=True)
            invert_palette = {v: k for k, v in palette.items()}
            mask_g = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
            for c, i in invert_palette.items():
                m = np.all(mask_rgb == np.array(c).reshape(1, 1, 3), axis=2)
                mask_g[m] = i
                break
            mask_a = Image.fromarray(mask_g, 'L')
            mask_a.save(out_path/f'{path.stem}.png')

class PrintStatusCallback():
    def __init__(self): pass
    def before_calc(self, slice_s, **kwargs): print(f"Process slice: {slice_s}")
    def status_calc(self, p ,li, **kwargs): print(f'[{p}|{li}]',end='\r')
    def after_calc (self, slice_s, last_slice,out_path, **kwargs):
        if last_slice != None:
            print(f"Finished")
        print("----")
        if slice_s == last_slice:
            print(f'Saved under: {out_path}')
            print('Count: ' , len(list(out_path.glob('*.png'))))