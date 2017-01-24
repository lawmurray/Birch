/**
 * @file
 */
#pragma once

#include "bi/random/Expirable.hpp"
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
template<class Variate, class Model, class Group = StackGroup>
class Random: public Expirable {
public:
  /**
   * Group type.
   */
  typedef Group group_type;

  /**
   * Pull function type.
   */
  typedef std::function<void()> pull_type;

  /**
   * Push function type.
   */
  typedef std::function<void()> push_type;

  /**
   * Constructor.
   */
  Random();

  /**
   * Constructor.
   */
  template<class Frame = EmptyFrame>
  Random(const Frame& frame = EmptyFrame(), const char* name = nullptr,
      const Group& group = Group());

  /**
   * Copy constructor.
   */
  Random(const Random<Variate,Model,Group>& o);

  /**
   * Move constructor.
   */
  Random(Random<Variate,Model,Group> && o);

  /**
   * Destructor.
   */
  virtual ~Random();

  /**
   * Assignment operator.
   */
  Random<Variate,Model,Group>& operator=(
      const Random<Variate,Model,Group>& o);

  /**
   * Assignent operator.
   */
  template<class Group1>
  Random<Variate,Model,Group>& operator=(
      const Random<Variate,Model,Group1>& o);

  /**
   * Assign variate.
   */
  Random<Variate,Model,Group>& operator=(const Variate& o);

  /**
   * Assign model.
   */
  Random<Variate,Model,Group>& operator=(const Model& o);

  /**
   * View operator.
   */
  template<class Frame, class View>
  Random<Variate,Model,Group> operator()(const Frame& frame,
      const View& view);

  /**
   * Cast to variate type.
   */
  operator Variate&();

  /**
   * Cast to filtered distribution type.
   */
  operator Model&();

  /**
   * Initialise.
   *
   * @param m Model.
   * @param pull Pull lambda.
   * @param push Push lambda.
   */
  void init(const Model& m, pull_type pull, push_type push);

  /**
   * Is the value missing?
   */
  bool isMissing() const;

  /**
   * Expire the random variable.
   */
  virtual void expire();

  /**
   * Group.
   */
  Group group;

  /**
   * Variate.
   */
  Variate x;

  /**
   * Filtered distribution of the random variable.
   */
  Model m;

private:
  /**
   * Pull lambda.
   */
  bi::model::Lambda<typename Group::child_group_type> pull;

  /**
   * Push lambda.
   */
  bi::model::Lambda<typename Group::child_group_type> push;

  /**
   * Position in random variable stack.
   */
  bi::model::Integer32<typename Group::child_group_type> pos;

  /**
   * Is the variate missing?
   */
  bi::model::Boolean<typename Group::child_group_type> missing;
};
}

#include "bi/random/RandomStack.hpp"

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>::Random() :
    pos(-1),
    missing(true) {
  //
}

template<class Variate, class Model, class Group>
template<class Frame>
bi::Random<Variate,Model,Group>::Random(const Frame& frame, const char* name,
    const Group& group) :
    group(childGroup(group, name)),
    x(frame, "x", childGroup(this->group, "x")),
    m(frame, "m", childGroup(this->group, "m")),
    pull(frame, "pull", childGroup(this->group, "pull")),
    push(frame, "push", childGroup(this->group, "push")),
    pos(-1),
    missing(true) {
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>::Random(const Random<Variate,Model,Group>& o) :
    group(o.group),
    x(o.x),
    m(o.m),
    pull(o.pull),
    push(o.push),
    pos(o.pos),
    missing(o.missing) {
  //
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>::Random(
    Random<Variate,Model,Group> && o) :
    group(o.group),
    x(o.x),
    m(o.m),
    pull(o.pull),
    push(o.push),
    pos(o.pos),
    missing(o.missing) {
  //
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>::~Random() {
  //
}

template<class Variate, class Model, class Group>
template<class Group1>
bi::Random<Variate,Model,Group>& bi::Random<Variate,Model,Group>::operator=(
    const Random<Variate,Model,Group1>& o) {
  x = o.x;
  m = o.m;
  pull = o.pull;
  push = o.push;
  missing = o.missing;
  pos = o.pos;

  return *this;
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>& bi::Random<Variate,Model,Group>::operator=(
    const Variate& o) {
  x = o;
  missing = false;

  return *this;
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>& bi::Random<Variate,Model,Group>::operator=(
    const Model& o) {
  /* pre-condition */
  assert(missing);

  m = o;
  return *this;
}

template<class Variate, class Model, class Group>
template<class Frame, class View>
bi::Random<Variate,Model,Group> bi::Random<Variate,Model,Group>::operator()(
    const Frame& frame, const View& view) {
  return Random<Variate,Model,Group>(*this, frame, view);
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>::operator Variate&() {
  if (missing) {
    randomStack.pop(pos);
  }
  assert(!missing);
  return x;
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>::operator Model&() {
  if (!missing) {
    throw std::bad_cast();
  }
  return m;
}

template<class Variate, class Model, class Group>
void bi::Random<Variate,Model,Group>::init(const Model& m, pull_type pull,
    push_type push) {
  this->m = m;
  this->pull = pull;
  this->push = push;

  if (!missing) {
    /* push immediately */
    static_cast<push_type>(push)();
  } else {
    /* lazy sampling */
    this->pos = randomStack.push(this);
  }
}

template<class Variate, class Model, class Group>
inline bool bi::Random<Variate,Model,Group>::isMissing() const {
  return missing;
}

template<class Variate, class Model, class Group>
void bi::Random<Variate,Model,Group>::expire() {
  /* pre-condition */
  assert(missing);

  static_cast<pull_type>(pull)();
  missing = false;
  static_cast<push_type>(push)();
}
