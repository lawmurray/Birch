/**
 * @file
 */
#pragma once

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
  Reference(const Target* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~Reference() = 0;

  /**
   * Does this reference resolve to a target that has previously captured
   * @p o?
   */
  template<class T>
  bool check(T* o);

  /**
   * Are these references canonically the same?
   */
  template<class T>
  bool canon(Reference<T>* o);

  /**
   * Target.
   */
  const Target* target;
};
}

template<class Target>
template<class T>
bool bi::Reference<Target>::check(T* o) {
  /* either this reference points to a parameter that captured something
   * equal to @p o earlier... */
  bool result = false /* *target->arg == *o*/;

  /* ...or this reference points to a parameter that captured another
   * parameter earlier, and o is a reference to that second parameter
   * to that parameter */
  const Reference<Target>* ref = dynamic_cast<const Reference<Target>*>(o);
  result = result || (ref && target->arg == ref->target);

  return result;
}

template<class Target>
template<class T>
bool bi::Reference<Target>::canon(Reference<T>* o) {
  return target == o->target;
}
