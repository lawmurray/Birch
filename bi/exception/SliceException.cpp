/**
 * @file
 */
#include "bi/exception/SliceException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::SliceException::SliceException(const Expression* o, const int typeSize,
    const int sliceSize) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: slice has incorrect number of dimensions\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << "\n";
  buf << "note: type has " << typeSize << " dimensions, slice has "
      << sliceSize << '\n';
  msg = base.str();
}
