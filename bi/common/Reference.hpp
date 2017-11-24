/**
 * @file
 */
#pragma once

#include <list>

namespace bi {
/**
 * Expression with a target (e.g. a reference).
 *
 * @ingroup birch_common
 */
template<class Target>
class Reference {
public:
  /**
   * Constructor.
   *
   * @param target Target.
   */
  Reference(Target* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~Reference() = 0;

  /**
   * Object to which the reference has been resolved.
   */
  Target* target;
};
}

template<class Target>
bi::Reference<Target>::Reference(Target* target) :
    target(target) {
  //
}

template<class Target>
bi::Reference<Target>::~Reference() {
  //
}
