/**
 * @file
 */
#include "bi/common/Location.hpp"

bi::Location::Location(File* file, const int firstLine, const int lastLine,
    const int firstCol, const int lastCol) :
    file(file), firstLine(firstLine), lastLine(lastLine), firstCol(firstCol), lastCol(
        lastCol) {
  //
}
