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
template<class Variate, class Model, class Group = MemoryGroup>
class Random: public virtual Expirable {
public:
  typedef Group group_type;
  typedef std::function<void()> pull_type;
  typedef std::function<void()> push_type;

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
  Random(const Random<Variate,Model,Group>& o) = default;

  /**
   * Move constructor.
   */
  Random(Random<Variate,Model,Group> && o) = default;

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
   * Variate copy assignment.
   */
  Random<Variate,Model,Group>& operator=(const Variate& o) {
    x = o;
  }

  /**
   * Variate move assignment.
   */
  Random<Variate,Model,Group>& operator=(Variate&& o) {
    x = o;
  }

  /**
   * Model copy assignment.
   */
  Random<Variate,Model,Group>& operator=(const Model& o) {
    m = o;
  }

  /**
   * Model move assignment.
   */
  Random<Variate,Model,Group>& operator=(Model&& o) {
    m = o;
  }

  /**
   * Cast to variate type.
   */
  operator Variate&();

  /**
   * Cast to filtered distribution type.
   */
  operator Model&();

  operator double&() {
    return static_cast<double&>(static_cast<PrimitiveValue<double,Group>&>(*this));
  }

  /**
   * View operator.
   */
  template<class Frame, class View>
  Random<Variate,Model,Group> operator()(const Frame& frame,
      const View& view) const;

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
  void init(const Model1& m, const pull_type& pull, const push_type& push);

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
  bi::model::Lambda<Group> pull;

  /**
   * Push lambda.
   */
  bi::model::Lambda<Group> push;

  /**
   * Position in random variable stack.
   */
  bi::model::Integer32<Group> pos;

  /**
   * Is the variate missing?
   */
  bi::model::Boolean<Group> missing;
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
    pull(nullptr, frame, "pull", childGroup(this->group, "pull")),
    push(nullptr, frame, "push", childGroup(this->group, "push")),
    pos(-1, frame, "pos", childGroup(this->group, "pos")),
    missing(true, frame, "missing", childGroup(this->group, "missing")) {
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>::Random(const EmptyFrame& frame,
    const char* name, const Group& group) :
    group(childGroup(group, name)),
    x(frame, "x", childGroup(this->group, "x")),
    m(frame, "m", childGroup(this->group, "m")),
    pull(nullptr, frame, "pull", childGroup(this->group, "pull")),
    push(nullptr, frame, "push", childGroup(this->group, "push")),
    pos(-1, frame, "pos", childGroup(this->group, "pos")),
    missing(true, frame, "missing", childGroup(this->group, "missing")) {
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
bi::Random<Variate,Model,Group>::~Random() {
  //
}

template<class Variate, class Model, class Group>
template<class Frame, class View>
bi::Random<Variate,Model,Group> bi::Random<Variate,Model,Group>::operator()(
    const Frame& frame, const View& view) const {
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
void bi::Random<Variate,Model,Group>::init(const Model1& m,
    const pull_type& pull, const push_type& push) {
  this->m = m;
  this->pull = pull;
  this->push = push;
  if (!isMissing()) {
    /* push immediately */
    static_cast<push_type>(this->push)();
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
  assert(isMissing());

  missing = false;
  static_cast<pull_type&>(pull)();
  static_cast<push_type&>(push)();
}
