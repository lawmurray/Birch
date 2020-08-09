/**
 * @file
 */
#include "bi/exception/Exception.hpp"

 bi::Exception::Exception() {
  //
}

bi::Exception::Exception(const std::string& msg) : msg(msg) {
  //
}
