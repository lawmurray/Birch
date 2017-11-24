/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"

namespace bi {
  namespace type {
/**
 * Root of the class hierarchy for all classes implemented in Birch.
 *
 * @ingroup libbirch
 */
class Object_: public Any {
  virtual Object_* clone() {
    return new (GC) Object_();
  }
};
  }
}
