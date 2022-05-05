import pathlib

from hatched import hatched

if __name__ == "__main__":
    image_path = pathlib.Path(__file__).parent / "skull.png"
    hatched.hatch(str(image_path), hatch_pitch=5, levels=(20, 100, 180), blur_radius=1)
