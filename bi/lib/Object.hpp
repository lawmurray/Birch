/**
 * @file
 */
#pragma once

#include "bi/lib/Any.hpp"

namespace bi {
  namespace type {
/**
 * Root of the class hierarchy for all classes implemented in Birch.
 *
 * @ingroup library
 */
class Object_: public Any {
  virtual Object_* clone() {
    return new (GC) Object_();
  }
};
  }
}
