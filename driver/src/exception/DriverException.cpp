/**
 * @file
 */
#include "src/exception/DriverException.hpp"

birch::DriverException::DriverException() {
  //
}

birch::DriverException::DriverException(const std::string& msg) {
  std::stringstream base;
  base << "error: " << msg << '\n';
  this->msg = base.str();
}
