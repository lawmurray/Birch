/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Expression with a target (e.g. a reference).
 *
 * @ingroup common
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

  /**
   * Alternative objects to which the reference might be resolved (e.g.
   * for member functions, those in base classes).
   */
  std::list<Target*> others;
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
