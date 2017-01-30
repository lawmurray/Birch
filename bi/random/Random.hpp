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
class Random {
public:
  typedef Group group_type;
  typedef Random<Variate,Model,Group> value_type;
  typedef std::function<void()> lambda_type;

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
  Random<Variate,Model,Group>& operator=(Random<Variate,Model,Group> && o) = default;

  /**
   * Variate copy assignment.
   */
  Random<Variate,Model,Group>& operator=(const Variate& o);

  /**
   * Variate move assignment.
   */
  Random<Variate,Model,Group>& operator=(Variate&& o);

  /**
   * Model copy assignment.
   */
  Random<Variate,Model,Group>& operator=(const Model& o);

  /**
   * Model move assignment.
   */
  Random<Variate,Model,Group>& operator=(Model&& o);

  /**
   * Cast to variate type.
   */
  operator typename Variate::value_type&();

  /**
   * Cast to filtered distribution type.
   */
  operator Model&();

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
  void init(const Model1& m, const lambda_type& pull,
      const lambda_type& push);

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
   * Filtered distribution of the random variable.
   */
  Model m;

private:
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
    pos(-1, frame, "pos", childGroup(this->group, "pos")),
    missing(true, frame, "missing", childGroup(this->group, "missing")) {
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>::Random(const EmptyFrame& frame,
    const char* name, const Group& group) :
    group(childGroup(group, name)),
    x(frame, "x", childGroup(this->group, "x")),
    m(frame, "m", childGroup(this->group, "m")),
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
bi::Random<Variate,Model,Group>& bi::Random<Variate,Model,Group>::operator=(
    const Variate& o) {
  x = o;
  missing = false;

  return *this;
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>& bi::Random<Variate,Model,Group>::operator=(
    Variate&& o) {
  x = o;
  missing = false;

  return *this;
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>& bi::Random<Variate,Model,Group>::operator=(
    const Model& o) {
  /* pre-condition */
  assert(isMissing());

  m = o;

  return *this;
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>& bi::Random<Variate,Model,Group>::operator=(
    Model&& o) {
  /* pre-condition */
  assert(isMissing());

  m = o;
  return *this;
}

template<class Variate, class Model, class Group>
bi::Random<Variate,Model,Group>::operator typename Variate::value_type&() {
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
    const lambda_type& pull, const lambda_type& push) {
  this->m = m;
  if (!isMissing()) {
    /* push immediately */
    push();
  } else {
    /* lazy sampling */
    this->pos = randomStack.push(pull, push);
  }
}

template<class Variate, class Model, class Group>
inline bool bi::Random<Variate,Model,Group>::isMissing() const {
  return missing;
}
