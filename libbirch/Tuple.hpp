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
  template<class Arg1, class ... Args1> friend class Tuple;
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
  template<class T = Arg>
  Tuple(const Arg& arg, Args ... args,
      typename std::enable_if_t<is_value<T>::value,int> = 0) :
      head(arg),
      tail(args...) {
    //
  }

  /**
   * Constructor.
   */
  template<class T = Arg>
  Tuple(Label* context, const Arg& arg, Args ... args,
      typename std::enable_if_t<is_value<T>::value,int> = 0) :
      head(arg),
      tail(context, args...) {
    //
  }

  /**
   * Constructor.
   */
  template<class T = Arg>
  Tuple(Label* context, const Arg& arg, Args ... args,
      typename std::enable_if_t<!is_value<T>::value,int> = 0) :
      head(context, arg),
      tail(context, args...) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class T = Arg>
  Tuple(const Tuple<Arg,Args...>& o,
      typename std::enable_if_t<is_value<T>::value,int> = 0) :
      head(o.head),
      tail(o.tail) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class T = Arg>
  Tuple(Label* context, const Tuple<Arg,Args...>& o,
      typename std::enable_if_t<is_value<T>::value,int> = 0) :
      head(o.head),
      tail(context, o.tail) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class T = Arg>
  Tuple(Label* context, const Tuple<Arg,Args...>& o,
      typename std::enable_if_t<!is_value<T>::value,int> = 0) :
      head(context, o.head),
      tail(context, o.tail) {
    //
  }

  /**
   * Generic copy assignment.
   */
  template<class Arg1, class ... Args1>
  Tuple<Arg,Args...>& operator=(const Tuple<Arg1,Args1...>& o) {
    head = o.head;
    tail = o.tail;
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class Arg1, class ... Args1>
  Tuple<Arg,Args...>& operator=(Tuple<Arg1,Args1...> && o) {
    head = std::move(o.head);
    tail = std::move(o.tail);
    return *this;
  }

  /**
   * Freeze.
   */
  void freeze();

  /**
   * Thaw.
   */
  void thaw(Label* label);

  /**
   * Finish.
   */
  void finish();

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
  template<class Arg1, class ... Args1> friend class Tuple;
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
  template<class T = Arg>
  Tuple(const Arg& arg,
      typename std::enable_if_t<is_value<T>::value,int> = 0) :
      head(arg) {
    //
  }

  /**
   * Constructor.
   */
  template<class T = Arg>
  Tuple(Label* context, const Arg& arg,
      typename std::enable_if_t<is_value<T>::value,int> = 0) :
      head(arg) {
    //
  }

  /**
   * Constructor.
   */
  template<class T = Arg>
  Tuple(Label* context, const Arg& arg,
      typename std::enable_if_t<!is_value<T>::value,int> = 0) :
      head(context, arg) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class T = Arg>
  Tuple(const Tuple<Arg>& o,
      typename std::enable_if_t<is_value<T>::value,int> = 0) :
      head(o.head) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class T = Arg>
  Tuple(Label* context, const Tuple<Arg>& o,
      typename std::enable_if_t<is_value<T>::value,int> = 0) :
      head(o.head) {
    //
  }

  /**
   * Copy constructor.
   */
  template<class T = Arg>
  Tuple(Label* context, const Tuple<Arg>& o,
      typename std::enable_if_t<!is_value<T>::value,int> = 0) :
      head(context, o.head) {
    //
  }

  /**
   * Generic copy assignment.
   */
  template<class Arg1>
  Tuple<Arg>& operator=(const Tuple<Arg1>& o) {
    head = o.head;
    return *this;
  }

  /**
   * Generic move assignment.
   */
  template<class Arg1>
  Tuple<Arg>& operator=(Tuple<Arg1> && o) {
    head = std::move(o.head);
    return *this;
  }

  /**
   * Freeze.
   */
  void freeze();

  /**
   * Thaw.
   */
  void thaw(Label* label);

  /**
   * Finish.
   */
  void finish();

private:
  /**
   * First element.
   */
  Arg head;
};
}

#include "libbirch/type.hpp"

template<class Arg, class ... Args>
void libbirch::Tuple<Arg,Args...>::freeze() {
  libbirch::freeze(head);
  libbirch::freeze(tail);
}

template<class Arg, class ... Args>
void libbirch::Tuple<Arg,Args...>::thaw(Label* label) {
  libbirch::thaw(head, label);
  libbirch::thaw(tail, label);
}

template<class Arg, class ... Args>
void libbirch::Tuple<Arg,Args...>::finish() {
  libbirch::finish(head);
  libbirch::finish(tail);
}

template<class Arg>
void libbirch::Tuple<Arg>::freeze() {
  libbirch::freeze(head);
}

template<class Arg>
void libbirch::Tuple<Arg>::thaw(Label* label) {
  libbirch::thaw(head, label);
}

template<class Arg>
void libbirch::Tuple<Arg>::finish() {
  libbirch::finish(head);
}
