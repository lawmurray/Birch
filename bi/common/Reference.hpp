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
  bool check(const T* o) const;

  /**
   * Are these references canonically the same?
   */
  template<class T>
  bool canon(const Reference<T>* o) const;

  /**
   * Target.
   */
  const Target* target;
};
}

template<class Target>
inline bi::Reference<Target>::Reference(const Target* target) :
    target(target) {
  //
}

template<class Target>
inline bi::Reference<Target>::~Reference() {
  //
}

template<class Target>
template<class T>
inline bool bi::Reference<Target>::check(const T* o) const {
  /* either this reference points to a parameter that captured something
   * equal to @p o earlier... */
  bool result = *target->arg == *o;

  /* ...or this reference points to a parameter that captured another
   * parameter earlier, and o is a reference to that second parameter
   * to that parameter */
  const Reference<Target>* ref = dynamic_cast<const Reference<Target>*>(o);
  result = result || (ref && target->arg == ref->target);

  return result;
}

template<class Target>
template<class T>
inline bool bi::Reference<Target>::canon(const Reference<T>* o) const {
  return target == o->target;
}
