/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppHeaderGenerator.hpp"

namespace bi {
/**
 * Output stream for C++ header files.
 *
 * @ingroup io
 */
class hpp_ostream: public CppHeaderGenerator {
public:
  hpp_ostream(std::ostream& base, const int level = 0) :
    CppHeaderGenerator(base, level) {
    //
  }
};
}
