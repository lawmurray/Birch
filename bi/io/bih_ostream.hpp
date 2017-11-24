/**
 * @file
 */
#pragma once

#include "bi/io/bi_ostream.hpp"

namespace bi {
/**
 * Output stream for Birch header files.
 *
 * @ingroup birch_io
 */
class bih_ostream: public bi_ostream {
public:
  bih_ostream(std::ostream& base, const int level = 0) :
      bi_ostream(base, level, true) {
    //
  }
};
}
