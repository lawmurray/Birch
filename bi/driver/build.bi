import basic;

/**
 * Build the project.
 *
 *   - `--include-dir` : Add search directory for header files.
 *   - `--lib-dir` : Add search directory for library files.
 *   - `--share-dir` : Add search directory for data files.
 *
 * These three options are analogous to their counterparts for a C/C++
 * compiler, and specify the locations in which the Birch compiler should
 * search for headers (both Birch and C/C++ headers), installed libraries and
 * installed data files. They may be given multiple times to specify multiple
 * directories in the order in which they are to be searched.
 *
 * After searching these directories, the Birch compiler will search the
 * environment variables `BIRCH_INCLUDE_PATH`, `BIRCH_LIBRARY_PATH` and
 * `BIRCH_SHARE_PATH`, followed by the directories of the compiler's own
 * installation, followed by the system-wide locations `/usr/local/` and
 * `/usr/`.
 *
 *   - `--prefix` : Installation prefix (default platform-specific).
 *   - `--enable-std` / `--disable-std` : Enable/disable the standard library.
 *   - `--enable-warnings` / `--disable-warnings` : Enable/disable warnings.
 *   - `--enable-debug` / `--disable-debug` : Enable/disable debug mode.
 *   - `--enable-verbose` / `--disable-verbose` : Verbose mode.
 */
program build(
    include_dir:String,
    lib_dir:String,
    share_dir:String,
    prefix:String,
    warnings:Boolean <- true,
    debug:Boolean <- true,
    verbose:Boolean <- true);
