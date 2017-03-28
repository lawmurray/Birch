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
  typedef typename value_type<ResultType>::type value_type;
  typedef std::function<value_type()> forward_type;
  typedef std::function<double(const value_type&)> backward_type;

  /**
   * Constructor.
   */
  Lambda(const forward_type forward = []() -> value_type { assert(false); },
      const backward_type backward = [](const value_type&) -> double { assert(false); }) :
      forward(forward),
      backward(backward) {
        //
  }

  /**
   * Value constructor.
   */
  Lambda(const value_type& value) :
      forward([&]() -> value_type { return value; }),
      backward([&](const value_type& o) -> double { const_cast<value_type&>(value) = o; return 0.0; }) {
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
  double operator()(const value_type& o) const {
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
