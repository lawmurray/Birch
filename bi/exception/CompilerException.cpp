/**
 * @file
 */
#include "bi/exception/CompilerException.hpp"

bi::CompilerException::CompilerException() {
  //
}

bi::CompilerException::CompilerException(const std::string& msg) {
  std::stringstream base;
  base << "error: " << msg << '\n';
  this->msg = base.str();
}
