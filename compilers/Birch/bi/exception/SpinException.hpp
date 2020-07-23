/**
 * @file
 */
#pragma once

#include "bi/exception/Exception.hpp"
#include "bi/expression/Spin.hpp"

namespace bi {
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
