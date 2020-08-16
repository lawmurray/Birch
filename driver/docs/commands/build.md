    birch build [options]

Build the project.

### Basic options

  - `--enable-unity` / `--disable-unity` (default enabled): Enable/disable unity build. A unity build is typically faster from a clean state, but does not support incremental builds (i.e. a change to any file will trigger a full rebuild).
  - `--enable-debug` / `--disable-debug` (default enabled): Enable/disable debug mode. In debug mode, assertion checking is enabled and most compiler optimizations are disabled.
  - `--enable-warnings` / `--disable-warnings` (default enabled): Enable/disable compiler warnings.
  - `--enable-verbose` / `--disable-verbose` (default enabled): Show all compiler output.

### Advanced options

  - `--work-dir` (default `.`): The working directory. This can be used to build from a different directory to the package itself.
  - `--prefix` (default platform-specific): Installation prefix. This can be used to install files to different directories than the default. It works in the same way as the `--prefix` option given to `configure` scripts.
  - `--arch=[native|js|wasm]` (default `native`): Target architecture. Valid options are `native` for the architecture of the current machine, `js` for JavaScript or `wasm` for WebAssembly. The latter two require the Emscripten compiler.

The following three options are analogous to their counterparts for a C/C++ compiler, and specify the locations in which the Birch compiler should
search for headers (both Birch and C/C++ headers), installed libraries and
installed data files. They may be given multiple times to specify multiple
directories in the order in which they are to be searched.

  - `--include-dir` : Add search directory for header files.
  - `--lib-dir` : Add search directory for library files.
  - `--share-dir` : Add search directory for data files.

After searching these directories, the Birch compiler will search the environment variables `BIRCH_INCLUDE_PATH`, `BIRCH_LIBRARY_PATH` and `BIRCH_SHARE_PATH`, followed by the directories of the compiler's own installation, followed by the system-wide locations `/usr/local/` and
`/usr/`.


## Optimization options

C++ compiler options are controlled automatically by the driver. A GNU Autotools build system (`autoconf`, `automake`, `libtool`) is used internally, such that there is some scope to modify compiler flags using environment variables such as `CPPFLAGS` and `CXXFLAGS`.

The following Birch-specific options are available. The defaults have been set to be appropriate for most models. Small performance gains might be realized by tweaking these.

  - `--enable-memory-pool` / `--disable-memory-pool` (default enabled): Enable/ disable the memory pool allocator. This is typically a little faster than standard `malloc`/`realloc`/`free` but uses more memory overall.
  - `--enable-lazy-deep-clone` / `--disable-lazy-deep-clone` (default enabled): Enable/disable lazy deep clone instead of eager deep clone of objects.
  - `--clone-memo-initial-size=n` (default 16): Initial allocation size (number of entries) in memos used for clones. Must be a positive power of 2.
