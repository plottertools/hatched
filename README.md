# _hatched_

<img src="https://i.imgur.com/QLlBpNU.png" width=300 /> <img src="https://i.imgur.com/fRIrPV2.jpg" width=300 />

Library and [vpype](https://github.com/abey79/vpype) plug-in to convert images to plotter-friendly, hatched patterns.

Built with [OpenCV](https://github.com/skvark/opencv-python), [scikit-image](https://scikit-image.org),
[Shapely](https://github.com/Toblerity/Shapely), [matplotlib](https://matplotlib.org) and
[svgwrite](https://github.com/mozman/svgwrite). You can reach the author
[Drawingbots](https://drawingbots.net)'s [Discord server](https://discordapp.com/invite/XHP3dBg).


## Getting Started

### Using with `vpype`

Using `hatched` as a [vpype](https://github.com/abey79/vpype) plug-in is the easiest way to get started. See _vpype_'s [installation instructions](https://vpype.readthedocs.io/en/latest/install.html) for information on how to install _vpype_.

If *vpype* was installed using pipx, use the following command:

```bash
$ pipx inject vpype hatched
```

If *vpype* was installed using pip in a virtual environment, activate the virtual environment and use the following command:

```bash
$ pip install hatched
```

You can confirm that the installation was successful with the following command, which also happens to tell you all
you need to know to use `hatched`:

```bash
$ vpype hatched --help
Usage: vpype hatched [OPTIONS] FILENAME

  Generate hatched pattern from an image.

  The hatches generated are in the coordinate of the input image. For
  example, a 100x100px image with generate hatches whose bounding box
  coordinates are (0, 0, 100, 100). The `--scale` option, by resampling the
  input image, indirectly affects the generated bounding box. The `--pitch`
  parameter sets the densest hatching frequency,

Options:
  --levels INTEGER...             Pixel value of the 3 thresholds between
                                  black, dark, light and white zones (0-255).
  -s, --scale FLOAT               Scale factor to apply to the image size.
  -i, --interpolation [linear|nearest]
                                  Interpolation used for scaling.
  -b, --blur INTEGER              Blur radius to apply to the image before
                                  applying thresholds.
  -p, --pitch LENGTH              Hatching pitch for the densest zones. This
                                  option understands supported units.
  -x, --invert                    Invert the image (and levels) before
                                  applying thresholds.
  -c, --circular                  Use circular instead of diagonal hatches.
  -o, --center                    Origin of circles relative to the image size.
                                  For example, (0.5, 0.5) corresponds to the 
                                  center of the image.
  -a, --angle                     Angle for diagonal hatches (in degrees)
  -d, --show-plot                 Display the contours and resulting pattern
                                  using matplotlib.
  -l, --layer LAYER               Target layer or 'new'.
  --help                          Show this message and exit.
```

To create a SVG, combine the `hatched` command with the `write` command (check `vpype`'s documentation for more
information). Here is an example:

```bash
$ vpype hatched --levels 64 128 192 -s 0.5 -p 4 input.jpg layout a4 write output.svg
```


### Using `hatched` as a library

To play with _hatched_, you need to checkout the source and install the dependencies in a virtual environment, for
example with the following steps:

```bash
$ git clone https://github.com/plottertools/hatched.git
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
- `levels`: tuple of the n thresholds for different shades (0-255). The plugin only accepts 3 thresholds, but using as a library it accepts any number.
- `h_mirror`: apply horizontal mirror on the image if True
- `invert`: invert pixel value of the input image before processing (in this case, the level thresholds are inverted as well)
- `circular`: use circular hatching instead of diagonal
- `center`: relative position of cirles' center when using circular hatching
- `hatch_angle`: hatching angle for diagonal hatches (in degrees)
- `show_plot`: (default True) display contours and final results with matplotlib
- `save_svg`: (default True) controls whether or not an output SVG file is created 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The example image `skull.jpg` is licenced under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>
