/**
 * A file handle.
 */
type File;

READ:Integer <- 1;
WRITE:Integer <- 2;
APPEND:Integer <- 3;

/**
 * Open a file for reading.
 *
 *   - file : The file name.
 */
function fopen(file:String) -> File {
  return fopen(file, READ);
}

/**
 * Open a file.
 *
 *   - file : The file name.
 *   - mode : The mode, either `READ`, `WRITE`, or `APPEND`.
 */
function fopen(file:String, mode:Integer) -> File {
  assert mode == READ || mode == WRITE || mode == APPEND;
  s:String;
  if (mode == READ) {
    s <- "r";
  } else if (mode == WRITE) {
    s <- "w";
  } else if (mode == APPEND) {
    s <- "a";
  }
  cpp{{
  FILE* stream = ::fopen(file_.c_str(), s_.c_str());
  lockf(fileno(stream), F_LOCK, 0);
  return stream;
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
