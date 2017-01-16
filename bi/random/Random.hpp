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
   * Pull function type.
   */
  typedef std::function<void(Random<Variate,Model>&)> pull_type;

  /**
   * Push function type.
   */
  typedef std::function<void()> push_type;

  /**
   * Constructor.
   */
  Random();

  /**
   * Initialise.
   *
   * @param m Model.
   * @param pull Pull lambda.
   * @param push Push lambda.
   */
  void init(const Model& m, pull_type pull, push_type push);

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
   * Is the value missing?
   */
  bool isMissing() const;

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
   * Pull lambda.
   */
  pull_type pull;

  /**
   * Push lambda.
   */
  push_type push;

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
    pull_type pull, push_type push) {
  this->m = m;
  this->pull = pull;
  this->push = push;

  if (!missing) {
    /* push immediately */
    push();
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
inline bool bi::Random<Variate,Model>::isMissing() const {
  return missing;
}

template<class Variate, class Model>
void bi::Random<Variate,Model>::expire() {
  /* pre-condition */
  assert(missing);

  pull(*this);
  missing = false;
  push();
}
