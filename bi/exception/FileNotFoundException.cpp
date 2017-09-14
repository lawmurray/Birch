/**
 * @file
 */
#include "bi/exception/FileNotFoundException.hpp"

#include <sstream>

bi::FileNotFoundException::FileNotFoundException(const std::string& name) {
  std::stringstream base;
  base << "error: file '" << name << "' not found\n";
  msg = base.str();
}
