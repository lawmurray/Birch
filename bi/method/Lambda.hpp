/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Lambda function.
 */
template<class Type>
class Lambda {
public:
  typedef std::function<Type()> forward_type;
  typedef std::function<void(const Type&)> backward_type;

  /**
   * Constructor.
   */
  Lambda(const forward_type& forward = forward_type(),
      const backward_type& backward = backward_type()) :
      forward(forward),
      backward(backward),
      memoised(false) {
    //
  }

  /**
   * Evaluate forward lambda.
   */
  const Type& operator()() const {
    if (!memoised) {
      /* allow memoisation to occur while const, to keep invisible */
      auto self = const_cast<Lambda<Type>*>(this);
      self->value = forward();
      self->memoised = true;
    }
    return value;
  }

  /**
   * Evaluate backward lambda.
   */
  void operator()(const Type& o) const {
    backward(o);
  }

private:
  /**
   * Forward lambda.
   */
  forward_type forward;

  /**
   * Backward lambda.
   */
  backward_type backward;

  /**
   * Forward memoised value.
   */
  Type value;

  /**
   * Is the result of the forward lambda memoised?
   */
  bool memoised;
};
}
