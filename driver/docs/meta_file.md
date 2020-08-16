Each Birch project contains a `META.json` file providing a name, version and description of the project, as well as a manifest of its files and dependencies. The file may contain the following keys.

  - `name`
    : The name of the project (as a string).

  - `version`
    : The version of the project (as a string).

  - `description`
    : A one-sentence description of the project (as a string).

  - `manifest`
    : An object containing the following keys, listing the files to be included in the project.

    - `source`
      : A list of source files (as strings) to be compiled. These typically have a `.bi` file extension. If C/C++ sources are included, they may have `.c` or `.cpp` file extensions.

    - `header`
      : A list of header files (as strings) to be installed. This is typically only used if C/C++ sources are included, in which case these files are likely to have `.h` or `.hpp` file extensions.

    - `data`
      : A list of data files (as strings) to be installed. These are typically input files provided by the project.

    - `other`
      : A list of other files (as strings) to be distributed but not installed. These are typically meta files, such as `README.md`, `LICENSE`, and `META.json` itself.

  - `require`
    : An object containing the following keys, describing the dependencies of the project. Their presence is checked during the build of the project.

    - `package`
      : A list of other Birch packages (as strings).

    - `header`
      : A list of external C/C++ header files (as strings).

    - `library`
      : A list of external libraries (as strings).

    - `program`
      : A list of external programs (as strings).
