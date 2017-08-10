/**
 * @file
 */
#pragma once

#include "bi/io/cpp_ostream.hpp"

namespace bi {
/**
 * Output stream for C++ source files.
 *
 * @ingroup compiler_io
 */
class hpp_ostream: public cpp_ostream {
public:
  hpp_ostream(std::ostream& base, const int level = 0) :
      cpp_ostream(base, level, true) {
    //
  }
};
}
