from hatched import hatched


if __name__ == "__main__":
    hatched.make_hatch_svg("skull.png", hatch_pitch=7, levels=(20, 100, 180), blur_radius=1)
