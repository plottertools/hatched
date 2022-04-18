import math
import os
import random
from typing import Any, Iterable, Tuple, Union

import cv2
import matplotlib.collections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import shapely.ops
import svgwrite as svgwrite
from shapely.geometry import LinearRing, MultiLineString, Polygon
from skimage import measure


def _build_circular_hatch(
    delta: float, offset: float, w: int, h: int, center: Tuple[float, float] = (0.5, 0.5)
):
    center_x = w * center[0]
    center_y = h * center[1]

    ls = []
    # If center_x or center_y > 1, ensure the full image is covered with lines
    max_radius = max(math.sqrt(w**2 + h**2), math.sqrt(center_x**2 + center_y**2))

    for r in np.arange(offset, max_radius, delta):
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


def _build_diagonal_hatch(delta: float, offset: float, w: int, h: int, angle: float = 45):
    # Keep angle between 0 and 180
    angle = angle % 180
    # Convert angle to rads
    angle_rad = angle * math.pi / 180

    lines = []
    # Draw vertical lines
    if angle == 90:
        for i in np.arange(offset, w + 1, delta):
            start = (i, 0)
            stop = (i, h)
            lines.append([start, stop])

    # Draw horizontal lines
    elif angle == 0:
        for j in np.arange(offset, h + 1, delta):
            start = (0, j)
            stop = (w, j)
            lines.append([start, stop])

    elif angle < 90:
        for i in np.arange(offset, h / math.tan(angle_rad) + w + 1, delta):
            j = abs(i * math.tan(angle_rad))

            if i <= w:
                start = (i, 0)
            else:
                start = (w, (i - w) * j / i)

            if j <= h:
                stop = (0, j)
            else:
                stop = ((j - h) * i / j, h)

            lines.append([start, stop])

    else:
        for i in np.arange(h / math.tan(angle_rad) + offset, w + 1, delta):
            j = abs((w - i) * math.tan(math.pi - angle_rad))

            if i >= 0:
                start = (i, 0)
            else:
                start = (0, -i * j / (w - i))

            if j >= h:
                stop = (w - (j - h) * (w - i) / j, h)
            else:
                stop = (w, j)

            lines.append([start, stop])
    return lines


def _plot_poly(geom, colspec=""):
    plt.plot(*geom.exterior.xy, colspec)
    for i in geom.interiors:
        plt.plot(*i.xy, colspec)


def _plot_geom(geom, colspec=""):
    if geom.geom_type == "Polygon":
        _plot_poly(geom, colspec)
    elif geom.geom_type == "MultiPolygon":
        for p in geom:
            _plot_poly(p, colspec)


def _build_mask(cnt):
    lr = [LinearRing(p[:, [1, 0]]) for p in cnt if len(p) >= 4]

    mask = None
    for r in lr:
        if mask is None:
            mask = Polygon(r)
        else:
            if r.is_ccw:
                mask = mask.union(Polygon(r).buffer(0.5))
            else:
                mask = mask.difference(Polygon(r).buffer(-0.5))

    return mask


def _save_to_svg(file_path: str, w: int, h: int, vectors: Iterable[MultiLineString]) -> None:
    dwg = svgwrite.Drawing(file_path, size=(w, h), profile="tiny", debug=False)

    dwg.add(
        dwg.path(
            " ".join(
                " ".join(
                    ("M" + " L".join(f"{x},{y}" for x, y in ls.coords)) for ls in mls.geoms
                )
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
    hatch_pitch: float = 5.0,
    levels: Union[int, Tuple[int, ...]] = (64, 128, 192),
    circular: bool = False,
    center: Tuple[float, float] = (0.5, 0.5),
    invert: bool = False,
    hatch_angle: float = 45,
) -> Tuple[MultiLineString, Any, Any, Any]:

    if not isinstance(levels, Tuple):
        levels = (levels,)

    if invert:
        levels = tuple(255 - i for i in reversed(levels))

    h, w = img.shape
    n_levels = len(levels)

    # border for contours to be closed shapes
    r = np.zeros(shape=(img.shape[0] + 2, img.shape[1] + 2))
    r[1:-1, 1:-1] = img

    # Find contours at a constant value of 0.8
    contours = [measure.find_contours(r, levels[i]) for i in range(n_levels)]

    mls = [MultiLineString(np.empty(shape=(0, 2, 2))) for i in range(n_levels)]

    try:
        mask = [_build_mask(i) for i in contours[::-1]]

        # Spacing considers interleaved lines from different levels
        delta_factors = [2 ** (n_levels - 1)]
        delta_factors.extend([2 ** (n_levels - i) for i in range(1, n_levels)])
        offset_factors = [0]
        offset_factors.extend([2 ** (n_levels - i - 1) for i in range(1, n_levels)])

        if circular:
            lines = [
                _build_circular_hatch(
                    delta_factors[i] * hatch_pitch,
                    offset_factors[i] * hatch_pitch,
                    w,
                    h,
                    center=center,
                )
                for i in range(len(levels))
            ]
        else:
            # correct offset to ensure desired distance between hatches
            if hatch_angle % 180 != 0:
                hatch_pitch /= math.sin((hatch_angle % 180) * math.pi / 180)

            lines = [
                _build_diagonal_hatch(
                    delta_factors[i] * hatch_pitch,
                    offset_factors[i] * hatch_pitch,
                    w,
                    h,
                    angle=hatch_angle,
                )
                for i in range(n_levels)
            ]

        frame = Polygon([(3, 3), (w - 6, 3), (w - 6, h - 6), (3, h - 6)])

        mls_ = [
            MultiLineString(MultiLineString(lines[i]).difference(mask[i]).intersection(frame))
            for i in range(n_levels)
        ]

        mls = [
            MultiLineString(shapely.ops.linemerge([i for i in mls_[j].geoms]))
            for j in range(n_levels)
        ]

    except Exception as exc:
        print(f"Error: {exc}")

    all_lines = []
    for i in range(n_levels):
        all_lines.extend([ls for ls in mls[i].geoms])

    return (MultiLineString(all_lines), *contours)


def hatch(
    file_path: str,
    hatch_pitch: float = 5,
    levels: Union[int, Tuple[int, ...]] = (64, 128, 192),
    blur_radius: int = 10,
    image_scale: float = 1.0,
    interpolation: int = cv2.INTER_LINEAR,
    h_mirror: bool = False,
    invert: bool = False,
    circular: bool = False,
    center: Tuple[float, float] = (0.5, 0.5),
    hatch_angle: float = 45,
    show_plot: bool = True,
    save_svg: bool = True,
) -> MultiLineString:

    """
    Create hatched shading vector for an image, display it and save it to svg.
    :param file_path: input image path
    :param hatch_pitch: hatching pitch in pixel (correspond to the densest possible hatching)
    :param levels: pixel values of the threshold levels for different shades from black to white (0-255)
    :param blur_radius: blurring radius to apply on the input image (0 to disable)
    :param image_scale: scale factor to apply on the image before processing
    :param interpolation: interpolation to apply for scaling (typically either
        `cv2.INTER_LINEAR` or `cv2.INTER_NEAREST`)
    :param h_mirror: apply horizontal mirror on the image if True
    :param invert: invert pixel value of the input image before processing (in this case, the
        level thresholds are inverted as well)
    :param circular: use circular hatching instead of diagonal
    :param center: relative x and y position for the center of circles when using circular
        hatching. Defaults to (0.5, 0.5) corresponding to the center of the image
    :param hatch_angle: angle that defines hatching inclination (degrees)
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

    mls, *cnts = _build_hatch(
        img,
        hatch_pitch=hatch_pitch,
        levels=levels,
        invert=invert,
        circular=circular,
        center=center,
        hatch_angle=hatch_angle,
    )

    if save_svg:
        # save vector data to svg file
        _save_to_svg(
            os.path.splitext(file_path)[0] + ".svg", img.shape[0], img.shape[1], [mls]
        )

    # Plot everything
    # ===============

    if show_plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap=plt.cm.gray)

        # noinspection PyShadowingNames
        def plot_cnt(contours, spec):
            for cnt in contours:
                plt.plot(cnt[:, 1], cnt[:, 0], spec, linewidth=2)

        colors = list(mcolors.BASE_COLORS.keys())
        for i, cnt in enumerate(cnts):
            plot_cnt(cnt, f"{colors[i]}-")

        plt.subplot(1, 2, 2)

        if invert:
            plt.gca().set_facecolor((0, 0, 0))
            color = (1, 1, 1)
        else:
            color = (0, 0, 0)

        plt.gca().add_collection(
            matplotlib.collections.LineCollection(
                (ls.coords for ls in mls.geoms), color=color, lw=0.3
            )
        )

        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])

        plt.show()

    return mls
