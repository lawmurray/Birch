/**
 * Check the file structure of the project for possible issues. This makes no
 * modifications to the project, but will output warnings for possible issues
 * such as:
 *
 *   - files listed in the `MANIFEST` file that do not exist,
 *   - files of recognisable types that exist but that are not listed in the
 *     `MANIFEST` file, and
 *   - standard project meta files that do not exist.
 */
program check();
