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
class Random: public Expirable {
public:
  /**
   * Constructor.
   */
  Random();

  /**
   * Initialise.
   *
   * @param m Model.
   * @param push Push expression.
   */
  void init(const Model& m, std::function<void()> push);

  /**
   * Assign to variate.
   */
  Random<Variate,Model>& operator=(const Variate& o);

  /**
   * Assign to model.
   */
  Random<Variate,Model>& operator=(const Model& o);

  /**
   * Cast to variate.
   */
  operator Variate&();

  /**
   * Cast to filtered distribution.
   */
  operator Model&();

  /**
   * Expire the random variable.
   */
  virtual void expire();

  /**
   * The variate.
   */
  Variate x;

  /**
   * The filtered distribution of the random variable.
   */
  Model m;

private:
  /**
   * Push function.
   */
  std::function<void(const Variate&)> push;

  /**
   * Position in random variable stack.
   */
  int pos;

  /**
   * Is the variate missing?
   */
  bool missing;
};
}

#include "bi/random/RandomStack.hpp"

template<class Variate, class Model>
bi::Random<Variate,Model>::Random() :
    pos(-1),
    missing(true) {
  //
}

template<class Variate, class Model>
void bi::Random<Variate,Model>::init(const Model& m,
    std::function<void()> push) {
  this->m = m;
  this->push = push;

  if (!missing) {
    /* push immediately */
    push(x);
  } else {
    /* lazy sampling */
    this->pos = randomStack.push(this);
  }
}

template<class Variate, class Model>
bi::Random<Variate,Model>& bi::Random<Variate,Model>::operator=(
    const Variate& o) {
  x = o;
  missing = false;
  return *this;
}

template<class Variate, class Model>
bi::Random<Variate,Model>& bi::Random<Variate,Model>::operator=(
    const Model& o) {
  /* pre-condition */
  assert(missing);

  m = o;
  return *this;
}

template<class Variate, class Model>
bi::Random<Variate,Model>::operator Variate&() {
  if (missing) {
    randomStack.pop(pos);
  }
  assert(!missing);
  return x;
}

template<class Variate, class Model>
bi::Random<Variate,Model>::operator Model&() {
  /* pre-condition */
  assert(missing);

  return m;
}

template<class Variate, class Model>
void bi::Random<Variate,Model>::expire() {
  /* pre-condition */
  assert(missing);

  pull_(x, m);
  missing = false;
  push(x);
}
