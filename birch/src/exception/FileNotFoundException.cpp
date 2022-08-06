/**
 * @file
 */
#include "src/exception/FileNotFoundException.hpp"

birch::FileNotFoundException::FileNotFoundException(const std::string& name) {
  std::stringstream base;
  base << "error: file '" << name << "' not found\n";
  msg = base.str();
}
