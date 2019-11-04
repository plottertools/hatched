# hatched

Convert photos to plotter-friendly hatched pattern.

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The example image `skull.jpg` is licenced under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>
