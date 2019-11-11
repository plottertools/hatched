import math
import os
import random
from typing import Tuple, Iterable, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely.ops
import svgwrite as svgwrite
from shapely.geometry import asMultiLineString, Polygon, LinearRing, MultiLineString
from skimage import measure


def _build_circular_hatch(delta: float, offset: float, w: int, h: int):
    center_x = w / 2
    center_y = h / 2

    ls = []
    for r in np.arange(offset, math.sqrt(w * w + h * h), delta):
        # make a tiny circle as point in the center
        if r == 0:
            r = 0.001

        # compute a meaningful number of segment adapted to the circle's radius
        n = max(20, r)
        t = np.arange(0, 1, 1 / n)

        # A random phase is useful for circles that end up unmasked. If several such circles
        # start and stop at the same location, a disgraceful pattern will emerge when plotting.
        phase = random.random() * 2 * math.pi

        data = np.array(
            [
                center_x + r * np.cos(t * math.pi * 2 + phase),
                center_y + r * np.sin(t * math.pi * 2 + phase),
            ]
        ).T
        ls.append(LinearRing(data))

    mls = MultiLineString(ls)

    # Crop the circle to the final dimension
    p = Polygon([(0, 0), (w, 0), (w, h), (0, h)])
    return mls.intersection(p)


def _build_diagonal_hatch(delta: float, offset: float, w: int, h: int):
    lines = []
    for i in np.arange(offset, h + w + 1, delta):
        if i < w:
            start = (i, 0)
        else:
            start = (w, i - w)

        if i < h:
            stop = (0, i)
        else:
            stop = (i - h, h)

        lines.append([start, stop])
    return np.array(lines)


def _build_mask(cnt):
    lr = [LinearRing(p[:, [1, 0]]) for p in cnt if len(p) >= 4]
    mask = shapely.ops.unary_union([Polygon(r).buffer(0.5) for r in lr if r.is_ccw])
    mask = mask.difference(
        shapely.ops.unary_union([Polygon(r).buffer(-0.5) for r in lr if not r.is_ccw])
    )
    return mask


def _save_to_svg(file_path: str, w: int, h: int, vectors: Iterable[MultiLineString]) -> None:
    dwg = svgwrite.Drawing(file_path, size=(w, h), profile="tiny", debug=False)

    dwg.add(
        dwg.path(
            " ".join(
                " ".join(("M" + " L".join(f"{x},{y}" for x, y in ls.coords)) for ls in mls)
                for mls in vectors
            ),
            fill="none",
            stroke="black",
        )
    )

    dwg.save()


def _load_image(
    file_path: str,
    blur_radius: int = 10,
    image_scale: float = 1.0,
    interpolation: int = cv2.INTER_LINEAR,
    h_mirror: bool = False,
    invert: bool = False,
) -> np.ndarray:
    # Load the image, resize it and apply blur
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    scale_x = int(img.shape[1] * image_scale)
    scale_y = int(img.shape[0] * image_scale)
    img = cv2.resize(img, (scale_x, scale_y), interpolation=interpolation)
    if blur_radius > 0:
        img = cv2.blur(img, (blur_radius, blur_radius))

    if h_mirror:
        img = np.flip(img, axis=1)

    if invert:
        img = 255 - img

    return img


def _build_hatch(
    img: np.ndarray,
    hatch_pitch: float = 5.,
    levels: Tuple[int, int, int] = (64, 128, 192),
    circular: bool = False,
    invert: bool = False,
) -> Tuple[MultiLineString, Any, Any, Any]:
    if invert:
        levels = tuple(255 - i for i in reversed(levels))

    h, w = img.shape

    # border for contours to be closed shapes
    r = np.zeros(shape=(img.shape[0] + 2, img.shape[1] + 2))
    r[1:-1, 1:-1] = img

    # Find contours at a constant value of 0.8
    black_cnt = measure.find_contours(r, levels[0])
    dark_cnt = measure.find_contours(r, levels[1])
    light_cnt = measure.find_contours(r, levels[2])

    light_mls = asMultiLineString(np.empty(shape=(0, 2, 2)))
    dark_mls = asMultiLineString(np.empty(shape=(0, 2, 2)))
    black_mls = asMultiLineString(np.empty(shape=(0, 2, 2)))

    try:
        black_p = _build_mask(black_cnt)
        dark_p = _build_mask(dark_cnt)
        light_p = _build_mask(light_cnt)

        if circular:
            build_func = _build_circular_hatch
        else:
            build_func = _build_diagonal_hatch

        light_lines = build_func(4 * hatch_pitch, 0, w, h)
        dark_lines = build_func(4 * hatch_pitch, 2 * hatch_pitch, w, h)
        black_lines = build_func(2 * hatch_pitch, hatch_pitch, w, h)

        frame = Polygon([(3, 3), (w - 6, 3), (w - 6, h - 6), (3, h - 6)])

        light_mls = shapely.ops.linemerge(
            asMultiLineString(light_lines).difference(light_p).intersection(frame)
        )
        dark_mls = shapely.ops.linemerge(
            asMultiLineString(dark_lines).difference(dark_p).intersection(frame)
        )
        black_mls = shapely.ops.linemerge(
            asMultiLineString(black_lines).difference(black_p).intersection(frame)
        )
    except Exception as exc:
        print(f"Error: {exc}")

    return (
        MultiLineString(
            [ls for ls in light_mls] + [ls for ls in dark_mls] + [ls for ls in black_mls]
        ),
        black_cnt,
        dark_cnt,
        light_cnt,
    )


def hatch(
    file_path: str,
    hatch_pitch: float = 5,
    levels: Tuple[int, int, int] = (64, 128, 192),
    blur_radius: int = 10,
    image_scale: float = 1.0,
    interpolation: int = cv2.INTER_LINEAR,
    h_mirror: bool = False,
    invert: bool = False,
    circular: bool = False,
    show_plot: bool = True,
    save_svg: bool = True,
) -> MultiLineString:

    """
    Create hatched shading vector for an image, display it and save it to svg.
    :param file_path: input image path
    :param hatch_pitch: hatching pitch in pixel (correspond to the densest possible hatching)
    :param levels: pixel value of the 3 threshold between black, dark, light and white (0-255)
    :param blur_radius: blurring radius to apply on the input image (0 to disable)
    :param image_scale: scale factor to apply on the image before processing
    :param interpolation: interpolation to apply for scaling (typically either
        `cv2.INTER_LINEAR` or `cv2.INTER_NEAREST`)
    :param h_mirror: apply horizontal mirror on the image if True
    :param invert: invert pixel value of the input image before processing (in this case, the
        level thresholds are inverted as well)
    :param circular: use circular hatching instead of diagonal
    :param show_plot: display contours and final results with matplotlib
    :param save_svg: controls whether or not an output svg file is created
    :return: MultiLineString Shapely object of the resulting hatch pattern
    """

    img = _load_image(
        file_path=file_path,
        blur_radius=blur_radius,
        image_scale=image_scale,
        interpolation=interpolation,
        h_mirror=h_mirror,
        invert=invert,
    )

    mls, black_cnt, dark_cnt, light_cnt = _build_hatch(
        img, hatch_pitch=hatch_pitch, levels=levels, invert=invert, circular=circular
    )

    if save_svg:
        # save vector data to svg file
        _save_to_svg(
            os.path.splitext(file_path)[0] + ".svg", img.shape[0], img.shape[1], [mls]
        )

    # Plot everything
    # ===============

    if show_plot:
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap=plt.cm.gray)

        # noinspection PyShadowingNames
        def plot_cnt(contours, spec):
            for cnt in contours:
                plt.plot(cnt[:, 1], cnt[:, 0], spec, linewidth=2)

        plot_cnt(black_cnt, "b-")
        plot_cnt(dark_cnt, "g-")
        plot_cnt(light_cnt, "r-")

        plt.subplot(1, 2, 2)

        if invert:
            plt.gca().set_facecolor((0, 0, 0))
            spec = "w-"
        else:
            spec = "k-"

        for ls in mls:
            plt.plot(ls.xy[0], img.shape[1] - np.array(ls.xy[1]), spec, lw=0.3)

        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])

        plt.show()

    return mls
