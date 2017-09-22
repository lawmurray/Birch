/**
 * A file handle.
 */
type File;

/**
 * Open a file for reading.
 *
 *   - file : The file name.
 */
function fopen(file:String) -> File {
  return fopen(file, "r");
}

/**
 * Open a file.
 *
 *   - file : The file name.
 *   - mode : The mode, either `r` (read), `w` (write), `a` (append) or any
 *     other modes as in system `fopen`.
 */
function fopen(file:String, mode:String) -> File {
  cpp{{
  return ::fopen(file_.c_str(), mode_.c_str());
  }}
}

/**
 * Close a file.
 */
function fclose(file:File) {
  cpp{{
  ::fclose(file_);
  }}
}
