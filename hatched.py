import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely.ops
import svgwrite as svgwrite
from shapely.geometry import asMultiLineString, Polygon, LinearRing
from skimage import measure


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
    mask = shapely.ops.unary_union([Polygon(r) for r in lr if r.is_ccw and r.is_valid])
    mask = mask.difference(
        shapely.ops.unary_union([Polygon(r) for r in lr if not r.is_ccw and r.is_valid])
    )
    return mask


def make_hatch_svg(
    file_path, hatch_pitch=5, levels=(64, 128, 192), blur_radius=10, image_scale=1
):

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.blur(img, (blur_radius, blur_radius))
    img = cv2.resize(
        img, (int(img.shape[1] * image_scale), int(img.shape[0] * image_scale))
    )

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
        black_p = build_mask(black_cnt)
        dark_p = build_mask(dark_cnt)
        light_p = build_mask(light_cnt)

        light_lines = build_hatch(4 * hatch_pitch, 0, w, h)
        dark_lines = build_hatch(4 * hatch_pitch, 2 * hatch_pitch, w, h)
        black_lines = build_hatch(2 * hatch_pitch, hatch_pitch, w, h)

        frame = Polygon([(3, 3), (w - 6, 3), (w - 6, h - 6), (3, h - 6)])

        light_mls = (
            asMultiLineString(light_lines).difference(light_p).intersection(frame)
        )
        dark_mls = asMultiLineString(dark_lines).difference(dark_p).intersection(frame)
        black_mls = (
            asMultiLineString(black_lines).difference(black_p).intersection(frame)
        )
    except Exception as exc:
        print(f"Error: {exc}")

    if w < h:
        nx = 1
        ny = 2
    else:
        nx = 2
        ny = 1

    plt.subplot(nx, ny, 1)
    plt.imshow(img, cmap=plt.cm.gray)

    def plot_cnt(contours, spec):
        for cnt in contours:
            plt.plot(cnt[:, 1], cnt[:, 0], spec, linewidth=2)

    plot_cnt(black_cnt, "b-")
    plot_cnt(dark_cnt, "g-")
    plot_cnt(light_cnt, "r-")

    plt.subplot(nx, ny, 2)

    for mls in [light_mls, dark_mls, black_mls]:
        for ls in mls:
            plt.plot(ls.xy[0], h - np.array(ls.xy[1]), "k-", lw=0.3)

    plt.axis("equal")

    plt.show()

    dwg = svgwrite.Drawing(
        os.path.splitext(file_path)[0] + ".svg",
        size=(w, h),
        profile="tiny",
        debug=False,
    )

    dwg.add(
        dwg.path(
            " ".join(
                (
                    " ".join(
                        ("M" + " L".join(f"{x},{y}" for x, y in ls.coords))
                        for ls in mls
                    )
                )
                for mls in [light_mls, dark_mls, black_mls]
            ),
            fill="none",
            stroke="black",
        )
    )

    dwg.save()


if __name__ == "__main__":
    make_hatch_svg("skull.jpg", hatch_pitch=3, levels=(50, 100, 200), blur_radius=5)
    # make_hatch_svg("skull2.jpg", hatch_pitch=5, levels=(50, 140, 200), blur_radius=1)
    # make_hatch_svg("skull3.jpg", hatch_pitch=5, levels=(30, 100, 150), blur_radius=5)
    # make_hatch_svg("poodle.jpg", hatch_pitch=3, levels=(50, 80, 140), blur_radius=5)
