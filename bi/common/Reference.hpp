/**
 * @file
 */
#pragma once

#include <list>

namespace bi {
/**
 * Expression with a target (e.g. a reference).
 *
 * @ingroup compiler_common
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
   * Target. This is the most-specific definite resolution to a parameter.
   */
  Target* target;

  /**
   * All matches.
   */
  std::list<Target*> matches;
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
