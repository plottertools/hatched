import math
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely.ops
import svgwrite as svgwrite
from shapely.geometry import (
    asMultiLineString,
    Polygon,
    LinearRing,
    asLinearRing,
    MultiLineString,
)
from skimage import measure


def build_circular_hatch(delta, offset, w, h):
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
        phase = random.random.rand() * 2 * math.pi

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


def build_hatch(delta, offset, w, h):
    lines = []
    for i in range(offset, h + w + 1, delta):
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


def build_mask(cnt):
    lr = [LinearRing(p[:, [1, 0]]) for p in cnt if len(p) >= 4]
    mask = shapely.ops.unary_union([Polygon(r).buffer(0.5) for r in lr if r.is_ccw])
    mask = mask.difference(
        shapely.ops.unary_union([Polygon(r).buffer(-0.5) for r in lr if not r.is_ccw])
    )
    return mask


def make_hatch_svg(
    file_path,
    hatch_pitch=5,
    levels=(64, 128, 192),
    blur_radius=10,
    image_scale=1,
    h_mirror=False,
    invert=False,
    circular=False,
):

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.blur(img, (blur_radius, blur_radius))

    scale_x = int(img.shape[1] * image_scale)
    scale_y = int(img.shape[0] * image_scale)
    img = cv2.resize(img, (scale_x, scale_y))

    h, w = img.shape

    if h_mirror:
        img = np.flip(img, axis=1)

    if invert:
        img = 255 - img
        levels = tuple(255 - i for i in reversed(levels))

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
        black_p = build_mask(black_cnt)
        dark_p = build_mask(dark_cnt)
        light_p = build_mask(light_cnt)

        if circular:
            build_func = build_circular_hatch
        else:
            build_func = build_hatch

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

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap=plt.cm.gray)

    def plot_cnt(contours, spec):
        for cnt in contours:
            plt.plot(cnt[:, 1], cnt[:, 0], spec, linewidth=2)

    plot_cnt(black_cnt, "b-")
    plot_cnt(dark_cnt, "g-")
    plot_cnt(light_cnt, "r-")

    plt.subplot(1, 2, 2)

    if invert:
        # plt.style.use('dark_background')
        plt.gca().set_facecolor((0, 0, 0))
        spec = "w-"
    else:
        spec = "k-"

    for mls in [light_mls, dark_mls, black_mls]:
        for ls in mls:
            plt.plot(ls.xy[0], h - np.array(ls.xy[1]), spec, lw=0.3)

    # for ls in light_p.boundary:
    #     plt.plot(ls.xy[0], h - np.array(ls.xy[1]), "r-", lw=0.3)

    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])

    plt.show()

    dwg = svgwrite.Drawing(
        os.path.splitext(file_path)[0] + ".svg", size=(w, h), profile="tiny", debug=False
    )

    dwg.add(
        dwg.path(
            " ".join(
                " ".join(("M" + " L".join(f"{x},{y}" for x, y in ls.coords)) for ls in mls)
                for mls in [light_mls, dark_mls, black_mls]
            ),
            fill="none",
            stroke="black",
        )
    )

    dwg.save()


if __name__ == "__main__":
    # make_hatch_svg("skull.jpg", hatch_pitch=3, levels=(50, 100, 200), blur_radius=5)
    # make_hatch_svg("skull2.jpg", hatch_pitch=5, levels=(50, 140, 200), blur_radius=1)
    # make_hatch_svg("skull3.jpg", hatch_pitch=5, levels=(30, 100, 150), blur_radius=5)
    # make_hatch_svg("poodle.jpg", hatch_pitch=3, levels=(50, 80, 140), blur_radius=5)
    # make_hatch_svg(
    #     "fish_clean.jpg", hatch_pitch=3, levels=(80, 180, 240), blur_radius=7, h_mirror=True
    # )

    # make_hatch_svg(
    #     "IMG_6094.jpeg",
    #     hatch_pitch=7,
    #     #levels=(40, 80, 120),
    #     levels=(80, 80, 40),
    #     #levels=(40, 100, 100),
    #     blur_radius=15,
    #     image_scale=0.25,
    #     invert=True,
    # )

    # make_hatch_svg(
    #     "IMG_6094.jpeg",
    #     hatch_pitch=7,
    #     levels=(40, 80, 120),
    #     # levels=(80, 80, 40),
    #     # levels=(40, 100, 100),
    #     blur_radius=15,
    #     image_scale=0.25,
    # )

    # make_hatch_svg(
    #     "my_skull.png", hatch_pitch=7, levels=(30, 100, 180), blur_radius=1,
    # )

    # make_hatch_svg(
    #     "IMG_1328.jpg",
    #     hatch_pitch=4,
    #     levels=(30, 120, 230),
    #     # levels=(80, 80, 40),
    #     # levels=(40, 100, 100),
    #     blur_radius=2,
    #     image_scale=0.4,
    #     circular=True,
    #     #invert=True
    # )

    make_hatch_svg(
        "IMG_2355.png",
        hatch_pitch=4,
        levels=(25, 80, 170),
        blur_radius=2,
        image_scale=0.4,
        circular=True,
        # invert=True
    )
