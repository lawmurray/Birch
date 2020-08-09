/**
 * @file
 */
#include "bi/common/Location.hpp"

bi::Location::Location(File* file, const int firstLine, const int lastLine,
    const int firstCol, const int lastCol, const std::string& doc) :
    file(file),
    firstLine(firstLine),
    lastLine(lastLine),
    firstCol(firstCol),
    lastCol(lastCol),
    doc(doc) {
  //
}
