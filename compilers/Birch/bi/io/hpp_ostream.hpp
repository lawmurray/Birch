/**
 * @file
 */
#pragma once

#include "bi/io/cpp_ostream.hpp"

namespace bi {
/**
 * Output stream for C++ header files.
 *
 * @ingroup io
 */
class hpp_ostream: public cpp_ostream {
public:
  hpp_ostream(std::ostream& base, const std::string& unit,
      const int level = 0) :
      cpp_ostream(base, unit, level, true) {
    //
  }
};
}
