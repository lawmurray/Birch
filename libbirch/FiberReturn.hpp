/**
 * @file
 */
#pragma once

namespace libbirch {
/**
 * Fiber return value.
 *
 * @ingroup libbirch
 *
 * @tparam Return Return type.
 */
template<class Return, class Enable = void>
class FiberReturn {
public:
  using return_type = Return;

  /**
   * Constructor.
   */
  FiberReturn() {
    //
  }

  /**
   * Constructor.
   */
  FiberReturn(const return_type& returnValue) :
      returnValue(returnValue) {
    //
  }

  /**
   * Cast to the return value.
   */
  operator return_type() const {
    return returnValue.get();
  }

private:
  /**
   * Return value.
   */
  Optional<return_type> returnValue;
};

template<class Return>
class FiberReturn<Return,IS_VOID(Return)> {
public:
  using return_type = Return;
};
}
