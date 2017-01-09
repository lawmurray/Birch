/**
 * @file
 */
#pragma once

#include "bi/random/Expirable.hpp"

#include <functional>

namespace bi {
/**
 * Random variable.
 *
 * @tparam Variate Variate type.
 * @tparam Model Model type.
 */
template<class Variate, class Model>
class Random : public Expirable {
public:
  /**
   * Constructor.
   *
   * @param x Variate.
   * @param m Model.
   * @param push Push expression.
   */
  Random(Variate& x, const Model& m, std::function<void()> push);

  /**
   * Destructor.
   */
  ~Random();

  /**
   * Cast to variate.
   */
  operator Variate&();

  /**
   * Expire the random variable.
   */
  virtual void expire();

  Variate& x;
  Model m;

private:
  /**
   * Push function.
   */
  std::function<void()> push;

  /**
   * Position in random variable stack.
   */
  int pos;
};
}

#include "bi/random/RandomStack.hpp"

template<class Variate, class Model>
bi::Random<Variate,Model>::Random(Variate& x, const Model& m,
    std::function<void()> push) :
    x(x),
    m(m),
    push(push),
    pos(randomStack.push(this)){
  //
}

template<class Variate, class Model>
bi::Random<Variate,Model>::~Random() {
  if (!isExpired()) {
    randomStack.pop(pos);
  }
}

template<class Variate, class Model>
bi::Random<Variate,Model>::operator Variate&() {
  if (!isExpired()) {
    randomStack.pop(pos);
  }
  return x;
}

template<class Variate, class Model>
void bi::Random<Variate,Model>::expire() {
  Expirable::expire();
  pull_(x, m);
  push();
}
