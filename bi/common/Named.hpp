/**
 * @file
 */
#pragma once

#include "bi/common/Name.hpp"
#include "bi/primitive/shared_ptr.hpp"

namespace bi {
/**
 * Named object.
 *
 * @ingroup compiler_common
 */
class Named {
public:
  /**
   * Empty constructor.
   */
  Named();

  /**
   * Constructor.
   *
   * @param name Name.
   */
  Named(shared_ptr<Name> name);

  /**
   * Destructor.
   */
  virtual ~Named() = 0;

  /**
   * Name.
   */
  shared_ptr<Name> name;
};
}

inline bi::Named::Named() :
    name(new bi::Name()) {
  //
}

inline bi::Named::Named(shared_ptr<Name> name) :
    name(name) {
  //
}

inline bi::Named::~Named() {
  //
}
