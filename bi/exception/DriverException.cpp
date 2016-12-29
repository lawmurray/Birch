/**
 * @file
 */
#include "bi/exception/DriverException.hpp"

#include <sstream>

bi::DriverException::DriverException() {
  //
}

bi::DriverException::DriverException(const std::string& msg) {
  std::stringstream base;
  base << "error: " << msg << '\n';
  this->msg = base.str();
}
