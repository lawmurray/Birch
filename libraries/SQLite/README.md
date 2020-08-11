# SQLite package

Birch language wrapper for the [SQLite](https://www.sqlite.org/) database. Currently provides most basic API functionality as described in [An Introduction To The SQLite C/C++ Interface](https://www.sqlite.org/cintro.html).


## Installation

To build and install, use:

    birch build
    birch install

Requires:

  * [SQLite](https://www.sqlite.org/)
    
To install Cairo on macOS with HomeBrew:

    brew install sqlite3

or on Ubuntu:

    apt-get install libsqlite3-dev

    
## Usage

To use from another Birch package, add `SQLite` to the `require.package` item in its `META.json`.