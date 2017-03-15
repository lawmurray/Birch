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
  Lambda(const forward_type forward = forward_type(),
      const backward_type backward = backward_type()) :
      forward(forward),
      backward(backward),
      memoised(false) {
    //
  }

  /**
   * Value constructor.
   */
  Lambda(const value_type& value) :
      backward([&](const value_type& o) {this->value = o;}),
      value(value),
      memoised(true) {
    //
  }

  /**
   * Cast to value.
   */
  operator const value_type&() const {
    return (*this)();
  }

  /**
   * Evaluate forward lambda.
   */
  const value_type& operator()() const {
    if (!memoised) {
      /* allow memoisation to occur while const, to keep invisible */
      auto self = const_cast<Lambda<ResultType>*>(this);
      self->value = forward();
      self->memoised = true;
    }
    return value;
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

  /**
   * Forward memoised value.
   */
  ResultType value;

  /**
   * Is the result of the forward lambda memoised?
   */
  bool memoised;
};
}
