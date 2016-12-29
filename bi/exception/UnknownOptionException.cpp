/**
 * @file
 */
#include "bi/exception/UnknownOptionException.hpp"

#include <sstream>

bi::UnknownOptionException::UnknownOptionException(const std::string& option) {
  std::stringstream base;
  base << "error: unknown program option '" << option << "'\n";
  msg = base.str();
}
