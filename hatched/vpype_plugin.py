import logging

import click
import cv2
import hatched

import vpype as vp


@click.command("hatched")
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "--levels",
    nargs=3,
    type=int,
    default=(64, 128, 192),
    help="Pixel value of the 3 thresholds between black, dark, light and white zones (0-255).",
)
@click.option("-s", "--scale", default=1.0, help="Scale factor to apply to the image size.")
@click.option(
    "-i",
    "--interpolation",
    default="linear",
    type=click.Choice(["linear", "nearest"], case_sensitive=False),
    help="Interpolation used for scaling.",
)
@click.option(
    "-b",
    "--blur",
    default=0,
    type=int,
    help="Blur radius to apply to the image before applying thresholds.",
)
@click.option(
    "-p",
    "--pitch",
    default=5,
    type=vp.LengthType(),
    help="Hatching pitch for the densest zones. This option understands supported units.",
)
@click.option(
    "-x",
    "--invert",
    is_flag=True,
    help="Invert the image (and levels) before applying thresholds.",
)
@click.option(
    "-c", "--circular", is_flag=True, help="Use circular instead of diagonal hatches."
)
@click.option(
    "-a",
    "--angle",
    default=45,
    type=float,
    help="Hatching angle for diagonal hatches (in degrees)"
)
@click.option(
    "-d",
    "--show-plot",
    is_flag=True,
    help="Display the contours and resulting pattern using matplotlib.",
)
@vp.generator
def hatched_gen(
    filename: str,
    levels,
    scale: float,
    interpolation: str,
    blur: int,
    pitch: int,
    invert: bool,
    circular: bool,
    angle: float,
    show_plot: bool,
):
    """
    Generate hatched pattern from an image.

    The hatches generated are in the coordinate of the input image. For example, a 100x100px
    image with generate hatches whose bounding box coordinates are (0, 0, 100, 100). The
    `--scale` option, by resampling the input image, indirectly affects the generated bounding
    box. The `--pitch` parameter sets the densest hatching frequency,
    """
    logging.info(f"generating hatches from {filename}")

    interp = cv2.INTER_LINEAR
    if interpolation == "nearest":
        interp = cv2.INTER_NEAREST

    return vp.LineCollection(
        hatched.hatch(
            file_path=filename,
            levels=levels,
            image_scale=scale,
            interpolation=interp,
            blur_radius=blur,
            hatch_pitch=pitch,
            invert=invert,
            circular=circular,
            hatch_angle=angle,
            show_plot=show_plot,
            h_mirror=False,  # this is best handled by vpype
            save_svg=False,  # this is best handled by vpype
        )
    )


hatched_gen.help_group = "Plugins"
