/**
 * @file
 */
#pragma once

#include "src/exception/Exception.hpp"
#include "src/expression/Spin.hpp"

namespace birch {
/**
 * Spin outside fiber exception.
 *
 * @ingroup exception
 */
struct SpinException: public Exception {
  /**
   * Constructor.
   */
  SpinException(const Spin* o);
};
}
