/**
 * @file
 */
#pragma once

#include "bi/bi.hpp"
#include "bi/method/RandomInterface.hpp"
#include "bi/method/RandomState.hpp"

#include <functional>

namespace bi {
/**
 * Random variable.
 *
 * @tparam Variate Variate type.
 * @tparam Model Model type.
 *
 * This is implemented using code similar to that generated for models by
 * the compiler itself so that it can act rather like it was specified in
 * Birch code. Eventually it may be possible to implement it directly in
 * Birch.
 */
template<class Variate, class Model>
class Random: public virtual RandomInterface {
public:
  /**
   * Value type.
   */
  typedef typename value_type<Variate>::type value_type;

  /**
   * Backward function type.
   */
  typedef std::function<double(const value_type&)> observe_type;

  /**
   * Constructor.
   */
  Random();

  /**
   * Destructor.
   */
  virtual ~Random();

  /**
   * Value assigment.
   */
  Random<Variate,Model>& operator=(const value_type& o);

  /**
   * Cast to variate type.
   */
  operator value_type&();

  /**
   * Cast to variate type.
   */
  operator const value_type&() const;

  /**
   * Cast to model type.
   */
  explicit operator Model&();

  /**
   * Cast to model type.
   */
  explicit operator const Model&() const;

  /**
   * Initialise the random variable.
   */
  void init(const Model& m, const observe_type& observeFunction);

  /*
   * RandomInterface requirements.
   */
  virtual void simulate();
  virtual double observe();
  virtual RandomState getState() const;
  virtual void setState(const RandomState state);
  virtual int getId() const;
  virtual void setId(const int id);

  /**
   * Backward function.
   */
  observe_type observeFunction;

  /**
   * Variate.
   */
  Variate x;

  /**
   * Model.
   */
  Model m;

  /**
   * Random variable id, or -1 if this has not been assigned.
   */
  int id;

  /**
   * Random variable state, taking one of the values of the enum State.
   */
  RandomState state;
};
}

#include "bi/method/Method.hpp"

template<class Variate, class Model>
bi::Random<Variate,Model>::Random() :
    id(-1),
    state(MISSING) {
  //
}

template<class Variate, class Model>
bi::Random<Variate,Model>::~Random() {
  //
}

template<class Variate, class Model>
bi::Random<Variate,Model>& bi::Random<Variate,Model>::operator=(
    const value_type& o) {
  x = o;
  state = ASSIGNED;

  return *this;
}

template<class Variate, class Model>
bi::Random<Variate,Model>::operator value_type&() {
  if (id >= 0 && state == MISSING) {
    method->simulate(id);
  }
  return x;
}

template<class Variate, class Model>
bi::Random<Variate,Model>::operator const value_type&() const {
  if (id >= 0 && state == MISSING) {
    method->simulate(id);
  }
  return x;
}

template<class Variate, class Model>
bi::Random<Variate,Model>::operator Model&() {
  if (id >= 0 && state == MISSING) {
    return m;
  } else {
    throw std::bad_cast();
  }
}

template<class Variate, class Model>
bi::Random<Variate,Model>::operator const Model&() const {
  if (id >= 0 && state == MISSING) {
    return m;
  } else {
    throw std::bad_cast();
  }
}

template<class Variate, class Model>
void bi::Random<Variate,Model>::init(const Model& m,
    const observe_type& observeFunction) {
  this->m = m;
  this->observeFunction = observeFunction;
  this->id = method->add(this);
}

template<class Variate, class Model>
void bi::Random<Variate,Model>::simulate() {
  simulate_(x, m);
}

template<class Variate, class Model>
double bi::Random<Variate,Model>::observe() {
  return observeFunction(x);
}

template<class Variate, class Model>
int bi::Random<Variate,Model>::getId() const {
  return this->id;
}

template<class Variate, class Model>
void bi::Random<Variate,Model>::setId(const int id) {
  this->id = id;
}

template<class Variate, class Model>
bi::RandomState bi::Random<Variate,Model>::getState() const {
  return this->state;
}

template<class Variate, class Model>
void bi::Random<Variate,Model>::setState(const RandomState state) {
  this->state = state;
}
