/**
 * @file
 */
#include "bi/exception/MemoryException.hpp"

#include <sstream>

bi::MemoryException::MemoryException(const std::string& msg) {
  std::stringstream base;
  base << "error: " << msg << '\n';
  this->msg = base.str();
}
