# Cairo package

Birch language wrapper for the [Cairo](https://www.cairographics.org/) 2d graphics library. Currently provides a limited subset of the functionality of the library.


## Installation

Requires:

  * [Cairo](https://www.cairographics.org/)
    
To install Cairo on macOS with HomeBrew:

    brew install cairo

or on Ubuntu:

    apt-get install libcairo2-dev

To build and install, use:

    birch build
    birch install


## Usage

To use from another Birch package, first add `Birch.Cairo` to the `require.package` item in its `META.json`. This will add checks for package files during the build.

Basic usage then looks something like this:

    surface:Surface <- createPNG(file, width, height);
    cr:Context <- create(surface);

    /* paint a black background */
    cr.setSourceRGB(0.0, 0.0, 0.0);
    cr.paint();

    /* draw a white rectangle */
    cr.setSourceRGB(1.0, 1.0, 1.0);
    cr.rectangle(0.25*width, 0.25*height, 0.5*width, 0.5*height);
    cr.fill();
  
    /* clean up */
    cr.destroy();
    surface.destroy();

Usage idioms are mostly preserved from Cairo itself. See the Cairo documentation for more details, and the package documentation (or code) for the subset of functionality provided.
