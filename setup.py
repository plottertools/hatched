from setuptools import setup

with open("README.md") as f:
    readme = f.read()

setup(
    name="hatched",
    version="0.1.0",
    description="Convert images to plotter-friendly hatched patterns",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Antoine Beyeler",
    author_email="abeyeler@gmail.com",
    url="https://github.com/plottertools/hatched",
    packages=["hatched"],
    setup_requires=["wheel"],
    install_requires=[
        "click",
        "vpype>=1.10,<2.0",
        "scikit-image",
        "svgwrite",
        "shapely>=1.8",
        "matplotlib",
        "opencv-python",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Topic :: Multimedia :: Graphics",
        "Environment :: Plugins",
    ],
    entry_points="""
            [vpype.plugins]
            hatched=hatched.vpype_plugin:hatched_gen
        """,
)
