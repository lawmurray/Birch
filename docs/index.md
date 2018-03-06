# Getting Started

The `birch` driver program can be used to set up a Birch project. From within a (usually empty) directory, run

    birch init --name Example

to create a new project, replacing `Example` with the name of the project. This creates the standard files and subdirectories for a Birch project. It is recommended that you maintain this standard structure to make it easier to manage and distribute your project.

Now is a good time to set up version control, such as Git, with this initial set of files.

The standard structure consists of the subdirectories:

  * `bi/` for source code,
  * `build/` for build files,
  * `input/` for input files,
  * `output/` for output files.

and a number of other meta files in the base directory. The most important of these meta files is `META.json`, which contains meta information such as a name, version, and description of the project, and a manifest of source files. As you add files to the project, especially `.bi` source files in the `bi/` subdirectory, you should add them to the `manifest.source` list in `META.json` to include them in the build.

To check for possible issues with the structure, use:

    birch check

To build the project, use:

    birch build

To install the project (required to run), use:

    birch install

To run a program that is either part of the standard library or among the source files of the code, use

    birch example

replacing `example` with the name of the program. Arguments to the program may be given as command-line options, e.g.:

    birch example -n 10 --input-file data/input.json

When building, Birch will create a number of additional files in the current working directory. Most of these are created in a `build/` subdirectory, although some will appear in the root directory of the project. To delete all of these additional files, use:

    birch clean

Debugging mode is enabled by default, and this will (dramatically) slow down execution times. It is recommended that you keep debugging mode enabled when developing and testing code (perhaps on small problems), but disable it when running tested code (perhaps on serious problems):

    birch build --disable-debug

More information on the `init`, `check`, and `build` programs is available in the documentation of the standard library, and more information on writing your own programs is available in the language guide below.
