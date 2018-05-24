cpp{{
#include "boost/filesystem.hpp"
}}

READ:Integer <- 1;
WRITE:Integer <- 2;
APPEND:Integer <- 3;

/**
 * Create a directory.
 *
 * - path: Path of the directory.
 */
function mkdir(path:String) {
  cpp{{
  boost::filesystem::create_directories(path_);
  }}
}

/**
 * Open a file for reading.
 *
 * - path : Path of the file.
 *
 * Return: File handle.
 *
 * If `path` includes non-existing directory, that directory is created (if
 * possible). The file is locked for reading (if possible).
 */
function fopen(path:String) -> File {
  return fopen(path, READ);
}

/**
 * Open a file.
 *
 * - path : Path of the file.
 * - mode : The mode, either `READ`, `WRITE`, or `APPEND`.
 *
 * Return: File handle.
 *
 * If `path` includes non-existing directory, that directory is created (if
 * possible). The file is locked for reading or writing as appropriate (if
 * possible).
 */
function fopen(path:String, mode:Integer) -> File {
  assert mode == READ || mode == WRITE || mode == APPEND;
  s:String;
  if (mode == READ) {
    s <- "r";
  } else if (mode == WRITE) {
    s <- "w";
    cpp{{
    boost::filesystem::path path = path_;
    if (!path.parent_path().empty()) {
      boost::filesystem::create_directories(path.parent_path());
    }
    }}
  } else if (mode == APPEND) {
    s <- "a";
  }
  cpp{{
  FILE* stream = ::fopen(path_.c_str(), s_.c_str());
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
