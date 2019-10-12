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
  void freeze() {
    libbirch::freeze(head);
    libbirch::freeze(tail);
  }

  /**
   * Thaw.
   */
  void thaw(Label* label) {
    libbirch::thaw(head, label);
    libbirch::thaw(tail, label);
  }

  /**
   * Finish.
   */
  void finish() {
    libbirch::finish(head);
    libbirch::finish(tail);
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
  void freeze() {
    libbirch::freeze(head);
  }

  /**
   * Thaw.
   */
  void thaw(Label* label) {
    libbirch::thaw(head, label);
  }

  /**
   * Finish.
   */
  void finish() {
    libbirch::finish(head);
  }

private:
  /**
   * First element.
   */
  Arg head;
};

template<class Arg>
struct is_value<Tuple<Arg>> {
  static const bool value = is_value<Arg>::value;
};

template<class Arg, class ... Args>
struct is_value<Tuple<Arg,Args...>> {
  static const bool value = is_value<Arg>::value && is_value<Tuple<Args...>>::value;
};

template<class... Args>
void freeze(Tuple<Args...>& o) {
  o.freeze();
}

template<class... Args>
void thaw(Tuple<Args...>& o, LazyLabel* label) {
  o.thaw(label);
}

template<class... Args>
void finish(Tuple<Args...>& o) {
  o.finish();
}

}
