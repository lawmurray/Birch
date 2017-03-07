/**
 * @file
 */
#pragma once

#include "bi/common/Name.hpp"
#include "bi/primitive/shared_ptr.hpp"

namespace bi {
/**
 * Mangled object.
 *
 * @ingroup compiler_common
 */
class Mangled {
public:
  /**
   * Empty constructor.
   */
  Mangled();

  /**
   * Constructor.
   *
   * @param name Name.
   */
  Mangled(shared_ptr<Name> name);

  /**
   * Destructor.
   */
  virtual ~Mangled() = 0;

  /**
   * Name.
   */
  shared_ptr<Name> mangled;
};
}
