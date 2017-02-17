/**
 * @file
 */
#pragma once

#include "bi/method/RandomInterface.hpp"
#include "bi/method/Random.hpp"

#include <functional>

namespace bi {
/**
 * Wrapper around Random objects for lazy evaluation. Keeps the model and
 * backward function associated with a random variable between being
 * initialised and being instantiated.
 *
 * @tparam Variate Variate type.
 * @tparam Model Model type.
 * @tparam Group Group type.
 */
template<class Variate, class Model, class Group>
class RandomLazy: public virtual RandomInterface {
public:
  /**
   * Simulate function type.
   */
  typedef std::function<typename Variate::value_type(Model&)> simulate_type;

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
  RandomLazy(Random<Variate,Model,Group>& x, const Model& m,
      const simulate_type& simulate, const backward_type& backward);

  /**
   * Destructor.
   */
  virtual ~RandomLazy();

  virtual void simulate();
  virtual double backward();
  virtual RandomState getState() const;
  virtual void setState(const RandomState state);
  virtual int getId() const;
  virtual void setId(const int id);

//private:
  /**
   * Random variable. This will usually be a shallow copy of the random
   * variable to be lazily evaluated. In some cases, such as random variables
   * that are returned from functions and would otherwise be freed, this will
   * be the canonical version of that random variable.
   */
  Random<Variate,Model,Group> rv;

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

template<class Variate, class Model, class Group>
bi::RandomLazy<Variate,Model,Group>::RandomLazy(
    Random<Variate,Model,Group>& rv, const Model& m,
    const simulate_type& simulate, const backward_type& backward) :
    rv(rv, false),  // shallow copy of random variable
    m(m),
    simulateFunction(simulate),
    backwardFunction(backward) {
  //
}

template<class Variate, class Model, class Group>
bi::RandomLazy<Variate,Model,Group>::~RandomLazy() {
  //
}

template<class Variate, class Model, class Group>
void bi::RandomLazy<Variate,Model,Group>::simulate() {
  rv.x = simulateFunction(m);
}

template<class Variate, class Model, class Group>
double bi::RandomLazy<Variate,Model,Group>::backward() {
  return backwardFunction();
}

template<class Variate, class Model, class Group>
int bi::RandomLazy<Variate,Model,Group>::getId() const {
  return rv.id;
}

template<class Variate, class Model, class Group>
void bi::RandomLazy<Variate,Model,Group>::setId(const int id) {
  rv.id = id;
}

template<class Variate, class Model, class Group>
bi::RandomState bi::RandomLazy<Variate,Model,Group>::getState() const {
  return rv.state;
}

template<class Variate, class Model, class Group>
void bi::RandomLazy<Variate,Model,Group>::setState(const RandomState state) {
  rv.state = state;
}
