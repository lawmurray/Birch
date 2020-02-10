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
  FiberReturn(Label* context) {
    //
  }

  /**
   * Constructor.
   */
  FiberReturn(Label* context, const return_type& returnValue) :
      returnValue(context, returnValue) {
    //
  }

  /**
   * Copy constructor.
   */
  FiberReturn(Label* context, const FiberReturn& o) :
      returnValue(context, o.returnValue) {
    //
  }

  /**
   * Move constructor.
   */
  FiberReturn(Label* context, FiberReturn&& o) :
      returnValue(context, std::move(o.returnValue)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  FiberReturn(Label* context, Label* label, const FiberReturn& o) :
      returnValue(context, label, o.returnValue) {
    //
  }

  /**
   * Copy assignment.
   */
  FiberReturn& assign(Label* context, const FiberReturn& o) {
    returnValue.assign(context, o.returnValue);
    return *this;
  }

  /**
   * Move assignment.
   */
  FiberReturn& assign(Label* context, FiberReturn&& o) {
    returnValue.assign(context, std::move(o.returnValue));
    return *this;
  }

  /**
   * Cast to the return value.
   */
  operator return_type() const {
    return returnValue.get();
  }

  void freeze() const {
    freeze(returnValue);
  }

  void thaw(Label* label) const {
    thaw(returnValue, label);
  }

  void finish() const {
    finish(returnValue);
  }

private:
  /**
   * Return value.
   */
  Optional<return_type> returnValue;
};

template<class Return>
class FiberReturn<Return,IS_VOID(Return)> {
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
  FiberReturn(Label* context) {
    //
  }

  /**
   * Copy constructor.
   */
  FiberReturn(Label* context, const FiberReturn& o) {
    //
  }

  /**
   * Move constructor.
   */
  FiberReturn(Label* context, FiberReturn&& o) {
    //
  }

  /**
   * Deep copy constructor.
   */
  FiberReturn(Label* context, Label* label, const FiberReturn& o) {
    //
  }

  /**
   * Copy assignment.
   */
  FiberReturn& assign(Label* context, const FiberReturn& o) {
    return *this;
  }

  /**
   * Move assignment.
   */
  FiberReturn& assign(Label* context, FiberReturn&& o) {
    return *this;
  }

  void freeze() const {
    //
  }

  void thaw(Label* label) const {
    //
  }

  void finish() const {
    //
  }
};
}
