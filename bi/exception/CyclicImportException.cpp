/**
 * @file
 */
#include "bi/exception/CyclicImportException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::CyclicImportException::CyclicImportException(File* o) {
  std::stringstream base;
  bih_ostream buf(base);
  buf << o->path << ": error: cyclic import of file\n";
  msg = base.str();
}
