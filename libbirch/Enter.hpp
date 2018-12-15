/**
 * @file
 */
#pragma once

#include "libbirch/Memo.hpp"

namespace bi {
/**
 * Enter a context while this remains in scope.
 *
 * @ingroup libbirch
 */
class Enter {
public:
  /**
   * Constructor.
   *
   * @param context The context to enter.
   */
  Enter(Memo* context) : prevContext(cloneMemo.get()) {
    cloneMemo = context;
  }

  /**
   * Destructor.
   */
  ~Enter() {
    cloneMemo = prevContext;
  }

private:
  /**
   * The previous context, to restore once this is destroyed.
   */
  Memo* prevContext;
};
}
