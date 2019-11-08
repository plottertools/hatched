# _hatched_

Convert images to plotter-friendly hatched patterns.

Built with [OpenCV](https://github.com/skvark/opencv-python), [scikit-image](https://scikit-image.org),
[Shapely](https://github.com/Toblerity/Shapely), [matplotlib](https://matplotlib.org) and
[svgwrite](https://github.com/mozman/svgwrite).


## Getting Started

### Installation

To play with _hatched_, you need to checkout the source and install the dependencies in a virtual environment, for
example with the following steps:

```bash
$ git clone https://github.com/abey79/hatched.git
$ cd hatched
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Running the example

Example can then be run by executing the corresponding file:

```bash
$ cd examples
$ python skull.py
```

The processing result is displayed in a matplotlib window:

<img width="752" alt="image" src="https://user-images.githubusercontent.com/49431240/68111504-8a279700-feef-11e9-9205-c60f06cfb828.png">

A `skull.svg` file is also created with the output graphics.

## Usage

Call the function `hatched.hatch()` to process your image. It takes the following parameters:

- `file_path`: input image (most common format are accepted)
- `image_scale`: scale factor to apply to the image before processing
- `interpolation`: interpolation to apply for scaling (typically either `cv2.INTER_LINEAR` or `cv2.INTER_NEAREST`)
- `blur_radius`: blurring radius to apply on the input image (0 to disable)
- `hatch_pitch`: hatching pitch in pixel (corresponds to the densest possible hatching)
- `levels`: tuple of the 3 thresholds between black, dark, light and white (0-255)
- `h_mirror`: apply horizontal mirror on the image if True
- `invert`: invert pixel value of the input image before processing (in this case, the level thresholds are inverted as well)
- `circular`: use circular hatching instead of diagonal
- `show_plot`: (default True) display contours and final results with matplotlib
- `save_svg`: (default True) controls whether or not an output SVG file is created 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The example image `skull.jpg` is licenced under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>
