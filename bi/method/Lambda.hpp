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
      backward(backward) {
    //
  }

  /**
   * Evaluate forward lambda.
   */
  Type operator()() const {
    return forward();
  }

  /**
   * Evaluate backward lambda.
   */
  void operator()(const Type& o) const {
    return forward(o);
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
};
}
