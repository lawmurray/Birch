/**
 * @file
 */
#pragma once

#include <functional>

namespace bi {
/**
 * Random variable.
 *
 * @tparam Variate Variate type.
 * @tparam Model Model type.
 */
template<class Variate, class Model>
class rv {
public:
  /**
   * Constructor.
   *
   * @param x Location in which to write variate, once simulated.
   * @param expr The model, as a lambda function.
   */
  rv(Variate& x, std::function<Model()> expr);

  /**
   * Destructor.
   */
  ~rv();

  /**
   * Cast to variate.
   */
  operator Variate&();

  /**
   * Cast to model.
   */
  operator Model&();

private:
  /**
   * Variate.
   */
  Variate& x;

  /**
   * Expression evaluate to give model.
   */
  std::function<Model()> expr;

  /**
   * Model.
   */
  Model p;

  /**
   * Marginalise forward.
   */
  void marginalise();

  /**
   * Simulate.
   */
  void simulate();

  /**
   * Condition backward.
   */
  void condition();

  /**
   * Has the model been created?
   */
  bool marginalised;

  /**
   * Has the variate been simulated?
   */
  bool simulated;

  /**
   * Have dependencies been conditioned?
   */
  bool conditioned;
};
}

template<class Variate, class Model>
bi::rv<Variate,Model>::rv(Variate& x, std::function<Model()> expr) :
    x(x),
    expr(expr),
    p(expr()),
    marginalised(true),
    simulated(false),
    conditioned(false) {
  //
}

template<class Variate, class Model>
bi::rv<Variate,Model>::~rv() {
  if (!marginalised) {
    marginalise();
  }
  if (!simulated) {
    simulate();
  }
  if (!conditioned) {
    condition();
  }
}

template<class Variate, class Model>
bi::rv<Variate,Model>::operator Variate&() {
  if (!marginalised) {
    marginalise();
  }
  if (!simulated) {
    simulate();
  }
  return x;
}

template<class Variate, class Model>
bi::rv<Variate,Model>::operator Model&() {
  /* pre-condition */
  assert(!simulated); // model has expired once variate has simulated

  if (!marginalised) {
    marginalise();
  }
  return p;
}

template<class Variate, class Model>
void bi::rv<Variate,Model>::marginalise() {
  /* pre-condition */
  assert(!marginalised);

  //p = expr();
  marginalised = true;
}

template<class Variate, class Model>
void bi::rv<Variate,Model>::simulate() {
  /* pre-condition */
  assert(marginalised);
  assert(!simulated);

  x = sim_(p);
  simulated = true;
}

template<class Variate, class Model>
void bi::rv<Variate,Model>::condition() {
  /* pre-condition */
  assert(simulated);
  assert(!conditioned);

  //sim_(x, p);
  conditioned = true;
}
