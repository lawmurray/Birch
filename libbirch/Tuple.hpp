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
  Tuple() = default;
  Tuple(const Tuple&) = default;
  Tuple(Tuple&&) = default;
  Tuple& operator=(const Tuple&) = delete;
  Tuple& operator=(Tuple&&) = delete;

  /**
   * Constructor for value type.
   */
  template<class T = Arg, std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple(const Arg& arg, Args ... args) :
      head(arg),
      tail(args...) {
    //
  }

  /**
   * Constructor for value head type.
   */
  template<class T = Arg, std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple(Label* context, const Arg& arg, Args ... args) :
      head(arg),
      tail(context, args...) {
    //
  }

  /**
   * Constructor for non-value head type.
   */
  template<class T = Arg, std::enable_if_t<!is_value<T>::value,int> = 0>
  Tuple(Label* context, const Arg& arg, Args ... args) :
      head(context, arg),
      tail(context, args...) {
    //
  }

  /**
   * Copy constructor for value head type.
   */
  template<class T = Arg, std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple(Label* context, const Tuple& o) :
      head(o.head),
      tail(context, o.tail) {
    //
  }

  /**
   * Copy constructor for non-value head type.
   */
  template<class T = Arg, std::enable_if_t<!is_value<T>::value,int> = 0>
  Tuple(Label* context, const Tuple& o) :
      head(context, o.head),
      tail(context, o.tail) {
    //
  }

  /**
   * Move constructor for value head type.
   */
  template<class T = Arg, std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple(Label* context, Tuple&& o) :
      head(std::move(o.head)),
      tail(context, std::move(o.tail)) {
    //
  }

  /**
   * Move constructor for non-value head type.
   */
  template<class T = Arg, std::enable_if_t<!is_value<T>::value,int> = 0>
  Tuple(Label* context, Tuple&& o) :
      head(context, std::move(o.head)),
      tail(context, std::move(o.tail)) {
    //
  }

  /**
   * Deep copy constructor for value head type.
   */
  template<class T = Arg, std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(o.head),
      tail(context, label, o.tail) {
    //
  }

  /**
   * Deep copy constructor for non-value head type.
   */
  template<class T = Arg, std::enable_if_t<!is_value<T>::value,int> = 0>
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(context, label, o.head),
      tail(context, label, o.tail) {
    //
  }

  /**
   * Copy assignment for value type.
   */
  template<class Arg1, class ... Args1, class T = Tuple,
      std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple& assign(const Tuple<Arg1,Args1...>& o) {
    head = o.head;
    tail = o.tail;
    return *this;
  }

  /**
   * Copy assignment for value head type.
   */
  template<class Arg1, class ... Args1, class T = Arg,
      std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple& assign(Label* context, const Tuple<Arg1,Args1...>& o) {
    head = o.head;
    tail.assign(context, o.tail);
    return *this;
  }

  /**
   * Copy assignment for non-value head type.
   */
  template<class Arg1, class ... Args1, class T = Arg,
      std::enable_if_t<!is_value<T>::value,int> = 0>
  Tuple& assign(Label* context, const Tuple<Arg1,Args1...>& o) {
    head.assign(context, o.head);
    tail.assign(context, o.tail);
    return *this;
  }

  /**
   * Move assignment for value type.
   */
  template<class Arg1, class ... Args1, class T = Tuple,
      std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple& assign(Tuple<Arg1,Args1...>&& o) {
    head = std::move(o.head);
    tail = std::move(o.tail);
    return *this;
  }

  /**
   * Move assignment for value head type.
   */
  template<class Arg1, class ... Args1, class T = Arg,
      std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple& assign(Label* context, Tuple<Arg1,Args1...>&& o) {
    head = std::move(o.head);
    tail.assign(context, std::move(o.tail));
    return *this;
  }

  /**
   * Move assignment for non-value head type.
   */
  template<class Arg1, class ... Args1, class T = Arg,
      std::enable_if_t<!is_value<T>::value,int> = 0>
  Tuple& assign(Label* context, Tuple<Arg1,Args1...>&& o) {
    head.assign(context, std::move(o.head));
    tail.assign(context, std::move(o.tail));
    return *this;
  }

  /**
   * Copy assignment operator for value type.
   */
  template<class Arg1, class ... Args1, class T = Tuple,
      std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple& operator=(const Tuple<Arg1,Args1...>& o) {
    return assign(o);
  }

  /**
   * Move assignment operator for value type.
   */
  template<class Arg1, class ... Args1, class T = Tuple,
      std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple& operator=(Tuple<Arg1,Args1...>&& o) {
    return assign(std::move(o));
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
  template<class Arg1, class... Args1> friend class Tuple;
public:
  Tuple() = default;
  Tuple(const Tuple&) = default;
  Tuple(Tuple&&) = default;
  Tuple& operator=(const Tuple&) = delete;
  Tuple& operator=(Tuple&&) = delete;

  /**
   * Constructor for value type.
   */
  template<class T = Arg, std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple(const Arg& arg) :
      head(arg) {
    //
  }

  /**
   * Constructor for value type.
   */
  template<class T = Arg, std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple(Label* context, const Arg& arg) :
      head(arg) {
    //
  }

  /**
   * Constructor for non-value type.
   */
  template<class T = Arg, std::enable_if_t<!is_value<T>::value,int> = 0>
  Tuple(Label* context, const Arg& arg) :
      head(context, arg) {
    //
  }

  /**
   * Copy constructor for value type.
   */
  template<class T = Arg, std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple(const Tuple& o) :
      head(o.head) {
    //
  }

  /**
   * Copy constructor for value type.
   */
  template<class T = Arg, std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple(Label* context, const Tuple& o) :
      head(o.head) {
    //
  }

  /**
   * Copy constructor for non-value type.
   */
  template<class T = Arg, std::enable_if_t<!is_value<T>::value,int> = 0>
  Tuple(Label* context, const Tuple& o) :
      head(context, o.head) {
    //
  }

  /**
   * Move constructor for value type.
   */
  template<class T = Arg, std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple(Tuple&& o) :
      head(std::move(o.head)) {
    //
  }

  /**
   * Move constructor for value type.
   */
  template<class T = Arg, std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple(Label* context, Tuple&& o) :
      head(std::move(o.head)) {
    //
  }

  /**
   * Move constructor for non-value type.
   */
  template<class T = Arg, std::enable_if_t<!is_value<T>::value,int> = 0>
  Tuple(Label* context, Tuple&& o) :
      head(context, std::move(o.head)) {
    //
  }

  /**
   * Deep copy constructor for value type.
   */
  template<class T = Arg, std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(o.head) {
    //
  }

  /**
   * Deep copy constructor for non-value type.
   */
  template<class T = Arg, std::enable_if_t<!is_value<T>::value,int> = 0>
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(context, label, o.head) {
    //
  }

  /**
   * Copy assignment for value type.
   */
  template<class Arg1, class T = Arg,
      std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple& assign(const Tuple<Arg1>& o) {
    head = o.head;
    return *this;
  }

  /**
   * Copy assignment for non-value type.
   */
  template<class Arg1, class T = Arg,
      std::enable_if_t<!is_value<T>::value,int> = 0>
  Tuple& assign(Label* context, const Tuple<Arg1>& o) {
    head.assign(context, o.head);
    return *this;
  }

  /**
   * Move assignment for value type.
   */
  template<class Arg1, class T = Arg,
      std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple& assign(Tuple<Arg1>&& o) {
    head = std::move(o.head);
    return *this;
  }

  /**
   * Move assignment for non-value type.
   */
  template<class Arg1, class T = Arg,
      std::enable_if_t<!is_value<T>::value,int> = 0>
  Tuple& assign(Label* context, Tuple<Arg1>&& o) {
    head.assign(context, std::move(o.head));
    return *this;
  }

  /**
   * Copy assignment operator for value type.
   */
  template<class Arg1, class T = Tuple,
      std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple& operator=(const Tuple<Arg1>& o) {
    return assign(o);
  }

  /**
   * Move assignment operator for value type.
   */
  template<class Arg1, class T = Tuple,
      std::enable_if_t<is_value<T>::value,int> = 0>
  Tuple& operator=(Tuple<Arg1>&& o) {
    return assign(std::move(o));
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
