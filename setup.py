from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license_text = f.read()

setup(
    name="hatched",
    version="0.0.1",
    description="Convert images to plotter-friendly hatched patterns",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Antoine Beyeler",
    author_email="abeyeler@gmail.com",
    url="https://github.com/abey79/hatched",
    license=license_text,
    packages=find_packages(exclude=("examples", "tests")),
    install_requires=[
        "scikit-image",
        "svgwrite",
        "shapely",
        "numpy",
        "matplotlib",
        "opencv-python",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
