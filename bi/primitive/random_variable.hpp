/**
 * @file
 */
#pragma once

#include "bi/bi.hpp"

#include <functional>

namespace bi {
/**
 * Random variable.
 *
 * @tparam Variate Variate type.
 * @tparam Model Model type.
 * @tparam Group Group type.
 *
 * This is implemented using code similar to that generated for models by
 * the compiler itself so that it can act rather like it was specified in
 * Birch code. Eventually it may be possible to implement it directly in
 * Birch, but for now generics are required.
 */
template<class Variate, class Model, class Group = MemoryGroup>
class random_variable {
public:
  typedef Group group_type;
  typedef random_variable<Variate,Model,Group> value_type;

  /**
   * Constructor.
   */
  template<class Tail, class Head>
  random_variable(const NonemptyFrame<Tail,Head>& frame, const char* name =
      nullptr, const Group& group = Group());

  /**
   * Constructor.
   */
  random_variable(const EmptyFrame& frame = EmptyFrame(), const char* name =
      nullptr, const Group& group = Group());

  /**
   * Copy constructor.
   */
  random_variable(const random_variable<Variate,Model,Group>& o) = default;

  /**
   * Move constructor.
   */
  random_variable(random_variable<Variate,Model,Group> && o) = default;

  /**
   * View constructor.
   */
  template<class Frame, class View>
  random_variable(const random_variable<Variate,Model,Group>& o,
      const Frame& frame, const View& view);

  /**
   * Destructor.
   */
  virtual ~random_variable();

  /**
   * Copy assignment.
   */
  random_variable<Variate,Model,Group>& operator=(
      const random_variable<Variate,Model,Group>& o) = default;

  /**
   * Move assignment.
   */
  random_variable<Variate,Model,Group>& operator=(
      random_variable<Variate,Model,Group> && o) = default;

  /**
   * Variate copy assignment.
   */
  random_variable<Variate,Model,Group>& operator=(const Variate& o);

  /**
   * Variate move assignment.
   */
  random_variable<Variate,Model,Group>& operator=(Variate&& o);

  /**
   * Cast to variate type.
   */
  operator typename Variate::value_type&();

  /**
   * Cast to model type.
   */
  explicit operator Model&();

  /**
   * View operator.
   */
  template<class Frame, class View>
  random_variable<Variate,Model,Group> operator()(const Frame& frame,
      const View& view) const;

  /**
   * Initialise the random variable.
   */
  template<class SimulateType, class BackwardType>
  void init(const Model& m, const SimulateType& simulate,
      const BackwardType& backward);

  /**
   * Is the value missing?
   */
  bool isMissing() const;

  /**
   * Group.
   */
  Group group;

  /**
   * Variate.
   */
  Variate x;

  /**
   * Random variable state. This is either < -1 for variate and missing
   * value, -1 for variate and known value, or >= 0 for model and missing
   * value (in which case it gives the unique non-negative id of the variable
   * as assigned by the method).
   */
  bi::model::Integer32<Group> state;

private:
  static const int VARIATE_MISSING;
  static const int VARIATE_KNOWN;
};
}

#include "bi/method/Method.hpp"
#include "bi/primitive/random_canonical_impl.hpp"

template<class Variate, class Model, class Group>
const int bi::random_variable<Variate,Model,Group>::VARIATE_MISSING = -2;

template<class Variate, class Model, class Group>
const int bi::random_variable<Variate,Model,Group>::VARIATE_KNOWN = -1;

template<class Variate, class Model, class Group>
template<class Tail, class Head>
bi::random_variable<Variate,Model,Group>::random_variable(
    const NonemptyFrame<Tail,Head>& frame, const char* name,
    const Group& group) :
    group(childGroup(group, name)),
    x(frame, "x", childGroup(this->group, "x")),
    state(VARIATE_MISSING, frame, "state", childGroup(this->group, "state")) {
  //
}

template<class Variate, class Model, class Group>
bi::random_variable<Variate,Model,Group>::random_variable(
    const EmptyFrame& frame, const char* name, const Group& group) :
    group(childGroup(group, name)),
    x(frame, "x", childGroup(this->group, "x")),
    state(VARIATE_MISSING, frame, "state", childGroup(this->group, "state")) {
  //
}

template<class Variate, class Model, class Group>
template<class Frame, class View>
bi::random_variable<Variate,Model,Group>::random_variable(
    const random_variable<Variate,Model,Group>& o, const Frame& frame,
    const View& view) :
    group(o.group),
    x(o.x, frame, view),
    state(o.state, frame, view) {
  //
}

template<class Variate, class Model, class Group>
bi::random_variable<Variate,Model,Group>::~random_variable() {
  //
}

template<class Variate, class Model, class Group>
template<class Frame, class View>
bi::random_variable<Variate,Model,Group> bi::random_variable<Variate,Model,
    Group>::operator()(const Frame& frame, const View& view) const {
  return random_variable<Variate,Model,Group>(*this, frame, view);
}

template<class Variate, class Model, class Group>
bi::random_variable<Variate,Model,Group>& bi::random_variable<Variate,Model,
    Group>::operator=(const Variate& o) {
  x = o;
  state = VARIATE_KNOWN;

  return *this;
}

template<class Variate, class Model, class Group>
bi::random_variable<Variate,Model,Group>& bi::random_variable<Variate,Model,
    Group>::operator=(Variate&& o) {
  x = o;
  state = VARIATE_KNOWN;

  return *this;
}

template<class Variate, class Model, class Group>
bi::random_variable<Variate,Model,Group>::operator typename Variate::value_type&() {
  if (isMissing()) {
    method->simulate(state);
    state = VARIATE_KNOWN;
  }
  return x;
}

template<class Variate, class Model, class Group>
bi::random_variable<Variate,Model,Group>::operator Model&() {
  if (!isMissing()) {
    throw std::bad_cast();
  } else {
    auto rv =
        reinterpret_cast<random_canonical_impl<Variate,Model>*>(method->get(
            state));
    return rv->m;
  }
}

template<class Variate, class Model, class Group>
template<class SimulateType, class BackwardType>
void bi::random_variable<Variate,Model,Group>::init(const Model& m,
    const SimulateType& simulate, const BackwardType& backward) {
  random_canonical* canonical = new random_canonical_impl<Variate,Model>(x, m,
      simulate, backward);
  state = method->add(canonical, state);
}

template<class Variate, class Model, class Group>
inline bool bi::random_variable<Variate,Model,Group>::isMissing() const {
  return state != VARIATE_KNOWN;
}
