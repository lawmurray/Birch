/**
 * @file
 */
#pragma once

#include "bi/bi.hpp"
#include "bi/method/RandomState.hpp"

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
 * Birch.
 */
template<class Variate, class Model, class Group = MemoryGroup>
class Random {
public:
  typedef Group group_type;
  typedef Random<Variate,Model,Group> value_type;

  /**
   * Constructor.
   */
  template<class Tail, class Head>
  Random(const NonemptyFrame<Tail,Head>& frame, const char* name = nullptr,
      const Group& group = Group()) :
      group(childGroup(group, name)),
      x(frame, "x", childGroup(this->group, "x")),
      id(frame, "id", childGroup(this->group, "id")),
      state(frame, "state", childGroup(this->group, "state")) {
    this->group.fill(id, -1, frame);
    this->group.fill(state, MISSING, frame);
  }

  /**
   * Constructor.
   */
  Random(const EmptyFrame& frame = EmptyFrame(), const char* name = nullptr,
      const Group& group = Group()) :
      group(childGroup(group, name)),
      x(frame, "x", childGroup(this->group, "x")),
      id(frame, "id", childGroup(this->group, "id")),
      state(frame, "state", childGroup(this->group, "state")) {
    this->group.fill(id, -1, frame);
    this->group.fill(state, MISSING, frame);
  }

  /**
   * View constructor.
   */
  template<class Frame, class View>
  Random(const Random<Variate,Model,Group>& o, const Frame& frame,
      const View& view) :
      group(o.group),
      x(o.x, frame, view),
      id(o.id, frame, view),
      state(o.state, frame, view) {
    //
  }

  /**
   * Copy constructor.
   */
  Random(const Random<Variate,Model,Group>& o) = default;

  /**
   * Move constructor.
   */
  Random(Random<Variate,Model,Group> && o) = default;

  /**
   * Generic copy constructor.
   */
  template<class Frame = EmptyFrame>
  Random(const Random<Variate,Model,Group>& o, const bool deep = true,
      const Frame& frame = EmptyFrame(), const char* name = nullptr,
      const MemoryGroup& group = MemoryGroup()) :
      x(o.x, deep, frame, name, group),
      id(o.id, deep, frame, name, group),
      state(o.state, deep, frame, name, group) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~Random();

  /**
   * Copy assignment.
   */
  Random<Variate,Model,Group>& operator=(const Random<Variate,Model,Group>& o) = default;

  /**
   * Move assignment.
   */
  Random<Variate,Model,Group>& operator=(Random<Variate,Model,Group>&& o) = default;

  /**
   * View operator.
   */
  template<class Frame, class View>
  Random<Variate,Model,Group> operator()(const Frame& frame,
      const View& view) const {
    return Random<Variate,Model,Group>(*this, frame, view);
  }

  /**
   * Variate copy assignment.
   */
  Random<Variate,Model,Group>& operator=(const typename Variate::value_type& o);

  /**
   * Cast to variate type.
   */
  operator typename Variate::value_type&();

  /**
   * Cast to model type.
   */
  explicit operator Model&();

  /**
   * Initialise the random variable.
   */
  template<class SimulateType, class BackwardType>
  void init(const Model& m, const SimulateType& simulate,
      const BackwardType& backward);

  /**
   * Group.
   */
  Group group;

  /**
   * Variate.
   */
  Variate x;

  /**
   * Random variable id, or -1 if this has not been assigned.
   */
  bi::model::Integer32<Group> id;

  /**
   * Random variable state, taking one of the values of the enum State.
   */
  bi::PrimitiveValue<RandomState,Group> state;
};
}

#include "bi/method/Method.hpp"
#include "bi/method/RandomLazy.hpp"

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>::~Random() {
  if (id >= 0) {
    /* random variable still persists in the method, extend its life by
     * giving ownership of it to the method */
    auto lazy = dynamic_cast<RandomLazy<Variate,Model,Group>*>(method->get(id));
    lazy->rv = std::move(*this);
  }
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>& bi::Random<Variate,Model,Group>::operator=(
    const typename Variate::value_type& o) {
  x = o;
  state = ASSIGNED;

  return *this;
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>::operator typename Variate::value_type&() {
  if (state == MISSING) {
    method->simulate(id);
  }
  assert(state != MISSING);
  return x;
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>::operator Model&() {
  if (id >= 0) {
    auto lazy = dynamic_cast<RandomLazy<Variate,Model,Group>*>(method->get(id));
    return lazy->m;
  } else {
    throw std::bad_cast();
  }
}

template<class Variate, class Model, class Group>
template<class SimulateType, class BackwardType>
void bi::Random<Variate,Model,Group>::init(const Model& m,
    const SimulateType& simulate, const BackwardType& backward) {
  auto rv = new RandomLazy<Variate,Model,Group>(*this, m, simulate, backward);
  id = method->add(rv);
}
