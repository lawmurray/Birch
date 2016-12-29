/**
 * @file
 */
#pragma once

namespace bi {
class File;

/**
 * Location within a file being parsed.
 *
 * @ingroup compiler_common
 */
struct Location {
  /**
   * Constructor.
   */
  Location(File* file = nullptr, const int firstLine = 0, const int lastLine =
      0, const int firstCol = 0, const int lastCol = 0);

  /**
   * File.
   */
  File* file;

  /*
   * Location in file.
   */
  int firstLine, lastLine, firstCol, lastCol;
};
}
