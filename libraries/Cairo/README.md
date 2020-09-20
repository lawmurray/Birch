# Cairo package

Birch language wrapper for the [Cairo](https://www.cairographics.org/) 2d graphics library. Currently provides a limited subset of the functionality of the library.


## License

Birch is open source software.

It is licensed under the Apache License, Version 2.0 (the "License"); you may not use it except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.


## Installation

Requires:

  * [Cairo](https://www.cairographics.org/)

To build and install, use:

    birch build
    birch install


## Usage

To use from another Birch package, first add `Cairo` to the `require.package` item in its `META.json`. This will add checks for package files during the build.

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
