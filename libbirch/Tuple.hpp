/**
 * @file
 */
#pragma once

namespace libbirch {
/**
 * Tuple.
 *
 * @ingroup libbirch
 */
template<class Arg, class ...Args>
class Tuple {
public:
  /**
   * Constructor.
   */
  Tuple() {
    //
  }

  /**
   * Constructor.
   */
  Tuple(const Arg& arg, Args ... args) :
      head(arg),
      tail(args...) {
    //
  }

  /**
   * Generic copy assignment.
   */
  template<class Arg1, class... Args1>
  Tuple<Arg,Args...>& operator=(const Tuple<Arg1,Args1...>& o) {
    head = o.head;
    tail = o.tail;
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class Arg1, class... Args1>
  Tuple<Arg,Args...>& operator=(Tuple<Arg1,Args1...>&& o) {
    head = std::move(o.head);
    tail = std::move(o.tail);
    return *this;
  }

private:
  /**
   * First element.
   */
  Arg head;

  /**
   * Remaining arguments.
   */
  Tuple<Args...> tail;
};

template<class Arg>
class Tuple<Arg> {
public:
  /**
   * Constructor.
   */
  Tuple(const Arg& arg) :
      head(arg) {
    //
  }

private:
  /**
   * First element.
   */
  Arg head;
};
}
