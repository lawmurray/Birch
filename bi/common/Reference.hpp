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
   * Target.
   */
  const Target* target;
};
}
