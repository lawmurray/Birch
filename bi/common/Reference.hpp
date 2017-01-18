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
   * Alternatives. This is the list of more-specific possible resolutions
   * that will be checked at runtime. It is only relevant for overloadable
   * parameters (e.g. functions).
   */
  std::list<Target*> alternatives;
};
}
