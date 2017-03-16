/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Lambda function.
 */
template<class ResultType>
class Lambda {
public:
  typedef typename ResultType::value_type value_type;
  typedef std::function<value_type()> forward_type;
  typedef std::function<void(const value_type&)> backward_type;

  /**
   * Constructor.
   */
  Lambda(const forward_type forward = []() -> value_type { assert(false); },
      const backward_type backward = [](const value_type&) { assert(false); }) :
      forward(forward),
      backward(backward) {
        //
  }

  /**
   * Value constructor.
   */
  Lambda(const value_type& value) :
      forward([&]() { return value; }),
      backward([&](const value_type& o) { const_cast<value_type&>(value) = o;}) {
    //
  }

  /**
   * Cast to value.
   */
  operator const value_type() const {
    return forward();
  }

  /**
   * Cast to value.
   */
  operator value_type() {
    return forward();
  }

  /**
   * Evaluate forward lambda.
   */
  const value_type operator()() const {
    return forward();
  }

  /**
   * Evaluate backward lambda.
   */
  void operator()(const value_type& o) const {
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
};
}
