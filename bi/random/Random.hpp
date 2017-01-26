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
class Random: public virtual Expirable {
public:
  typedef Group group_type;
  typedef std::function<void()> pull_type;
  typedef std::function<void()> push_type;

  template<class Group1>
  using regroup_type = Random<Variate,Model,Group1>;

  /**
   * Constructor.
   */
  template<class Tail, class Head>
  Random(const NonemptyFrame<Tail,Head>& frame, const char* name = nullptr,
      const Group& group = Group());

  /**
   * Constructor.
   */
  Random(const EmptyFrame& frame = EmptyFrame(), const char* name = nullptr,
      const Group& group = Group());

  /**
   * View constructor.
   */
  template<class Frame, class View>
  Random(const Random<Variate,Model,Group>& o, const Frame& frame,
      const View& view);

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
   * Generic assignent operator.
   */
  template<class Variate1, class Model1, class Group1>
  Random<Variate,Model,Group>& operator=(
      const Random<Variate1,Model1,Group1>& o);

  /**
   * Assign variate.
   */
  template<class Group1>
  Random<Variate,Model,Group>& operator=(
      const typename Variate::template regroup_type<Group1>& o);

  /**
   * Assign model.
   */
  template<class Group1>
  Random<Variate,Model,Group>& operator=(
      const typename Model::template regroup_type<Group1>& o);

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
   * @tparam Model1 Model type. Group may be different.
   *
   * @param m Model.
   * @param pull Pull lambda.
   * @param push Push lambda.
   */
  template<class Model1>
  void init(const Model1& m, pull_type pull, push_type push);

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
template<class Tail, class Head>
bi::Random<Variate,Model,Group>::Random(const NonemptyFrame<Tail,Head>& frame,
    const char* name, const Group& group) :
    group(childGroup(group, name)),
    x(frame, "x", childGroup(this->group, "x")),
    m(frame, "m", childGroup(this->group, "m")),
    pull(frame, "pull", childGroup(this->group, "pull")),
    push(frame, "push", childGroup(this->group, "push")),
    pos(-1, frame, "push", childGroup(this->group, "push")),
    missing(true, frame, "push", childGroup(this->group, "push")) {
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>::Random(const EmptyFrame& frame,
    const char* name, const Group& group) :
    group(childGroup(group, name)),
    x(frame, "x", childGroup(this->group, "x")),
    m(frame, "m", childGroup(this->group, "m")),
    pull(frame, "pull", childGroup(this->group, "pull")),
    push(frame, "push", childGroup(this->group, "push")),
    pos(-1, frame, "push", childGroup(this->group, "push")),
    missing(true, frame, "push", childGroup(this->group, "push")) {
}

template<class Variate, class Model, class Group>
template<class Frame, class View>
bi::Random<Variate,Model,Group>::Random(const Random<Variate,Model,Group>& o,
    const Frame& frame, const View& view) :
    group(o.group),
    x(o.x, frame, view),
    m(o.m, frame, view),
    pull(o.pull, frame, view),
    push(o.push, frame, view),
    pos(o.pos, frame, view),
    missing(o.missing, frame, view) {
  //
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
bi::Random<Variate,Model,Group>::Random(Random<Variate,Model,Group> && o) :
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
template<class Variate1, class Model1, class Group1>
bi::Random<Variate,Model,Group>& bi::Random<Variate,Model,Group>::operator=(
    const Random<Variate1,Model1,Group1>& o) {
  x = o.x;
  m = o.m;
  pull = o.pull;
  push = o.push;
  missing = o.missing;
  pos = o.pos;

  return *this;
}

template<class Variate, class Model, class Group>
template<class Group1>
bi::Random<Variate,Model,Group>& bi::Random<Variate,Model,Group>::operator=(
    const typename Variate::template regroup_type<Group1>& o) {
  x = o;
  missing = false;

  return *this;
}

template<class Variate, class Model, class Group>
template<class Group1>
bi::Random<Variate,Model,Group>& bi::Random<Variate,Model,Group>::operator=(
    const typename Model::template regroup_type<Group1>& o) {
  /* pre-condition */
  assert(isMissing());

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
  if (isMissing()) {
    randomStack.pop(pos);
  }
  assert(!isMissing());
  return x;
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>::operator Model&() {
  if (!isMissing()) {
    throw std::bad_cast();
  }
  return m;
}

template<class Variate, class Model, class Group>
template<class Model1>
void bi::Random<Variate,Model,Group>::init(const Model1& m, pull_type pull,
    push_type push) {
  this->m = m;
  this->pull = pull;
  this->push = push;

  if (!isMissing()) {
    /* push immediately */
    static_cast<push_type>(push)();
  } else {
    /* lazy sampling */
    this->pos = randomStack.push(this);
  }
}

template<class Variate, class Model, class Group>
inline bool bi::Random<Variate,Model,Group>::isMissing() const {
  return static_cast<unsigned char>(missing);
}

template<class Variate, class Model, class Group>
void bi::Random<Variate,Model,Group>::expire() {
  /* pre-condition */
  assert(isMissing());

  static_cast<pull_type>(pull)();
  missing = false;
  static_cast<push_type>(push)();
}
