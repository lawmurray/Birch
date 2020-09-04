/**
 * @file
 */
#pragma once

namespace birch {
class File;

/**
 * Location within a file being parsed.
 *
 * @ingroup common
 *
 * Location objects may be shared between multiple objects in the abstract
 * syntax tree, they need not be uniquely assigned.
 */
struct Location {
  /**
   * Constructor.
   *
   * @param file File.
   * @param firstLine Starting line number.
   * @param lastLine Ending line number.
   * @param firstCol Starting column number.
   * @param lastCol Ending column number.
   * @param doc Immediately preceding documentation comment, if any.
   */
  Location(File* file = nullptr, const int firstLine = 0, const int lastLine =
      0, const int firstCol = 0, const int lastCol = 0,
      const std::string& doc = "");

  /**
   * File.
   */
  File* file;

  /*
   * Location in file.
   */
  int firstLine, lastLine, firstCol, lastCol;

  /**
   * Preceding documentation comment.
   */
  std::string doc;
};
}
