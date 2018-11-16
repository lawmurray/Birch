/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/global.hpp"

namespace bi {
/**
 * Auxiliary class for switching the current memo. This may be used either
 * on the stack to switch the memo for the lifetime of a temporary variable:
 *
 *     Enter enter(memo);
 *
 * or inherited from a class to switch the memo during initialization,
 * explicitly calling exit() to restore the previous memo.
 *
 *     class A : public Enter {
 *       A() : ..., Enter(memo), ... {
 *         exit();
 *       }
 *     }
 */
class Enter {
public:
  /**
   * Constructor.
   *
   * @param memo The memo to enter.
   */
  Enter(Memo* memo) : prevMemo(memo) {
    std::swap(prevMemo, cloneMemo);
  }

  /**
   * Destructor.
   */
  ~Enter() {
    exit();
  }

  /**
   * Exit.
   */
  void exit() {
    std::swap(prevMemo, cloneMemo);
  }

private:
  /**
   * The previous memo.
   */
  Memo* prevMemo;
};
}
