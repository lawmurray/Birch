/**
 * @file
 */
#pragma once

#include <functional>

namespace bi {
/**
 * Canonical model associated with a random variable.
 *
 * @tparam Variate Variate type.
 * @tparam Model Model type.
 */
template<class Variate, class Model>
class random_canonical_impl: public random_canonical {
public:
  /**
   * Simulate function type.
   */
  typedef std::function<Variate(Model&)> simulate_type;

  /**
   * Backward function type.
   */
  typedef std::function<double()> backward_type;

  /**
   * Constructor.
   *
   * @param x Variate.
   * @param m Model.
   * @param simulate Simulate function.
   * @param backward Backward function.
   */
  random_canonical_impl(Variate& x, const Model& m,
      const simulate_type& simulate, const backward_type& backward);

  virtual void simulate();
  virtual double backward();

private:
  /**
   * Variate.
   */
  Variate x;

  /**
   * Model.
   */
  Model m;

  /**
   * Simulate function.
   */
  simulate_type simulateFunction;

  /**
   * Backward function.
   */
  backward_type backwardFunction;
};
}

template<class Variate, class Model>
bi::random_canonical_impl<Variate,Model>::random_canonical_impl(Variate& x,
    const Model& m, const simulate_type& simulate,
    const backward_type& backward) :
    x(x),  ///@todo Needs to be a view of the original, not a copy
    m(m),
    simulateFunction(simulate),
    backwardFunction(backward) {
  //
}

template<class Variate, class Model>
void bi::random_canonical_impl<Variate,Model>::simulate() {
  x = simulateFunction(m);
}

template<class Variate, class Model>
double bi::random_canonical_impl<Variate,Model>::backward() {
  return backwardFunction();
}
