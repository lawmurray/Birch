# SQLite package

Birch language wrapper for the [SQLite](https://www.sqlite.org/) database. Currently provides most basic API functionality as described in [An Introduction To The SQLite C/C++ Interface](https://www.sqlite.org/cintro.html).


## License

Birch is open source software.

It is licensed under the Apache License, Version 2.0 (the "License"); you may not use it except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.


## Installation

Requires:

  * [SQLite](https://www.sqlite.org/)

To build and install, use:

    birch build
    birch install


## Usage

To use from another Birch package, add `SQLite` to the `require.package` item in its `META.json`.
