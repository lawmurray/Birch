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
template<class Head, class ...Tail>
class Tuple {
  template<class Head1, class ... Tail1> friend class Tuple;
public:
  Tuple() = default;
  Tuple(const Tuple&) = default;
  Tuple(Tuple&&) = default;
  Tuple& operator=(const Tuple&) = delete;
  Tuple& operator=(Tuple&&) = delete;

  /**
   * Constructor for value type.
   */
  template<IS_VALUE(Head)>
  Tuple(const Head& head, Tail ... tail) :
      head(head),
      tail(tail...) {
    //
  }

  /**
   * Constructor for value head type.
   */
  template<IS_VALUE(Head)>
  Tuple(Label* context, const Head& head, Tail ... tail) :
      head(head),
      tail(context, tail...) {
    //
  }

  /**
   * Constructor for non-value head type.
   */
  template<IS_NOT_VALUE(Head)>
  Tuple(Label* context, const Head& head, Tail ... tail) :
      head(context, head),
      tail(context, tail...) {
    //
  }

  /**
   * Copy constructor for value head type.
   */
  template<IS_VALUE(Head)>
  Tuple(Label* context, const Tuple& o) :
      head(o.head),
      tail(context, o.tail) {
    //
  }

  /**
   * Copy constructor for non-value head type.
   */
  template<IS_NOT_VALUE(Head)>
  Tuple(Label* context, const Tuple& o) :
      head(context, o.head),
      tail(context, o.tail) {
    //
  }

  /**
   * Move constructor for value head type.
   */
  template<IS_VALUE(Head)>
  Tuple(Label* context, Tuple&& o) :
      head(std::move(o.head)),
      tail(context, std::move(o.tail)) {
    //
  }

  /**
   * Move constructor for non-value head type.
   */
  template<IS_NOT_VALUE(Head)>
  Tuple(Label* context, Tuple&& o) :
      head(context, std::move(o.head)),
      tail(context, std::move(o.tail)) {
    //
  }

  /**
   * Deep copy constructor for value head type.
   */
  template<IS_VALUE(Head)>
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(o.head),
      tail(context, label, o.tail) {
    //
  }

  /**
   * Deep copy constructor for non-value head type.
   */
  template<IS_NOT_VALUE(Head)>
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(context, label, o.head),
      tail(context, label, o.tail) {
    //
  }

  /**
   * Copy assignment for value type.
   */
  template<class Head1, class ... Tail1, IS_VALUE(Tuple)>
  Tuple& assign(const Tuple<Head1,Tail1...>& o) {
    head = o.head;
    tail = o.tail;
    return *this;
  }

  /**
   * Copy assignment for value head type.
   */
  template<class Head1, class ... Tail1, IS_VALUE(Head)>
  Tuple& assign(Label* context, const Tuple<Head1,Tail1...>& o) {
    head = o.head;
    tail.assign(context, o.tail);
    return *this;
  }

  /**
   * Copy assignment for non-value head type.
   */
  template<class Head1, class ... Tail1, IS_NOT_VALUE(Head)>
  Tuple& assign(Label* context, const Tuple<Head1,Tail1...>& o) {
    head.assign(context, o.head);
    tail.assign(context, o.tail);
    return *this;
  }

  /**
   * Move assignment for value type.
   */
  template<class Head1, class ... Tail1, IS_VALUE(Tuple)>
  Tuple& assign(Tuple<Head1,Tail1...>&& o) {
    head = std::move(o.head);
    tail = std::move(o.tail);
    return *this;
  }

  /**
   * Move assignment for value head type.
   */
  template<class Head1, class ... Tail1, IS_VALUE(Head)>
  Tuple& assign(Label* context, Tuple<Head1,Tail1...>&& o) {
    head = std::move(o.head);
    tail.assign(context, std::move(o.tail));
    return *this;
  }

  /**
   * Move assignment for non-value head type.
   */
  template<class Head1, class ... Tail1, IS_NOT_VALUE(Head)>
  Tuple& assign(Label* context, Tuple<Head1,Tail1...>&& o) {
    head.assign(context, std::move(o.head));
    tail.assign(context, std::move(o.tail));
    return *this;
  }

  /**
   * Copy assignment operator for value type.
   */
  template<class Head1, class ... Tail1, IS_VALUE(Tuple)>
  Tuple& operator=(const Tuple<Head1,Tail1...>& o) {
    return assign(o);
  }

  /**
   * Move assignment operator for value type.
   */
  template<class Head1, class ... Tail1, IS_VALUE(Tuple)>
  Tuple& operator=(Tuple<Head1,Tail1...>&& o) {
    return assign(std::move(o));
  }

  template<IS_VALUE(Head)>
  void freeze() {
    tail.freeze();
  }

  template<IS_NOT_VALUE(Head)>
  void freeze() {
    head.freeze();
    tail.freeze();
  }

  template<IS_VALUE(Head)>
  void thaw(Label* label) {
    tail.thaw(label);
  }

  template<IS_NOT_VALUE(Head)>
  void thaw(Label* label) {
    head.thaw(label);
    tail.thaw(label);
  }

  template<IS_VALUE(Head)>
  void finish() {
    tail.finish();
  }

  template<IS_NOT_VALUE(Head)>
  void finish() {
    head.finish();
    tail.finish();
  }

private:
  /**
   * First element.
   */
  Head head;

  /**
   * Remaining headuments.
   */
  Tuple<Tail...> tail;
};

template<class Head>
class Tuple<Head> {
  template<class Head1, class... Tail1> friend class Tuple;
public:
  Tuple() = default;
  Tuple(const Tuple&) = default;
  Tuple(Tuple&&) = default;
  Tuple& operator=(const Tuple&) = delete;
  Tuple& operator=(Tuple&&) = delete;

  /**
   * Constructor for value type.
   */
  template<IS_VALUE(Head)>
  Tuple(const Head& head) :
      head(head) {
    //
  }

  /**
   * Constructor for value type.
   */
  template<IS_VALUE(Head)>
  Tuple(Label* context, const Head& head) :
      head(head) {
    //
  }

  /**
   * Constructor for non-value type.
   */
  template<IS_NOT_VALUE(Head)>
  Tuple(Label* context, const Head& head) :
      head(context, head) {
    //
  }

  /**
   * Copy constructor for value type.
   */
  template<IS_VALUE(Head)>
  Tuple(const Tuple& o) :
      head(o.head) {
    //
  }

  /**
   * Copy constructor for value type.
   */
  template<IS_VALUE(Head)>
  Tuple(Label* context, const Tuple& o) :
      head(o.head) {
    //
  }

  /**
   * Copy constructor for non-value type.
   */
  template<IS_NOT_VALUE(Head)>
  Tuple(Label* context, const Tuple& o) :
      head(context, o.head) {
    //
  }

  /**
   * Move constructor for value type.
   */
  template<IS_VALUE(Head)>
  Tuple(Tuple&& o) :
      head(std::move(o.head)) {
    //
  }

  /**
   * Move constructor for value type.
   */
  template<IS_VALUE(Head)>
  Tuple(Label* context, Tuple&& o) :
      head(std::move(o.head)) {
    //
  }

  /**
   * Move constructor for non-value type.
   */
  template<IS_NOT_VALUE(Head)>
  Tuple(Label* context, Tuple&& o) :
      head(context, std::move(o.head)) {
    //
  }

  /**
   * Deep copy constructor for value type.
   */
  template<IS_VALUE(Head)>
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(o.head) {
    //
  }

  /**
   * Deep copy constructor for non-value type.
   */
  template<IS_NOT_VALUE(Head)>
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(context, label, o.head) {
    //
  }

  /**
   * Copy assignment for value type.
   */
  template<class Head1, IS_VALUE(Head)>
  Tuple& assign(const Tuple<Head1>& o) {
    head = o.head;
    return *this;
  }

  /**
   * Copy assignment for non-value type.
   */
  template<class Head1, IS_NOT_VALUE(Head)>
  Tuple& assign(Label* context, const Tuple<Head1>& o) {
    head.assign(context, o.head);
    return *this;
  }

  /**
   * Move assignment for value type.
   */
  template<class Head1, IS_VALUE(Head)>
  Tuple& assign(Tuple<Head1>&& o) {
    head = std::move(o.head);
    return *this;
  }

  /**
   * Move assignment for non-value type.
   */
  template<class Head1, IS_NOT_VALUE(Head)>
  Tuple& assign(Label* context, Tuple<Head1>&& o) {
    head.assign(context, std::move(o.head));
    return *this;
  }

  /**
   * Copy assignment operator for value type.
   */
  template<class Head1, IS_VALUE(Tuple)>
  Tuple& operator=(const Tuple<Head1>& o) {
    return assign(o);
  }

  /**
   * Move assignment operator for value type.
   */
  template<class Head1, IS_VALUE(Tuple)>
  Tuple& operator=(Tuple<Head1>&& o) {
    return assign(std::move(o));
  }

  template<IS_VALUE(Head)>
  void freeze() {
    //
  }

  template<IS_NOT_VALUE(Head)>
  void freeze() {
    head.freeze();
  }

  template<IS_VALUE(Head)>
  void thaw(Label* label) {
    //
  }

  template<IS_NOT_VALUE(Head)>
  void thaw(Label* label) {
    head.thaw(label);
  }

  template<IS_VALUE(Head)>
  void finish() {
    //
  }

  template<IS_NOT_VALUE(Head)>
  void finish() {
    head.finish();
  }

private:
  /**
   * First element.
   */
  Head head;
};

template<class Head>
struct is_value<Tuple<Head>> {
  static const bool value = is_value<Head>::value;
};

template<class Head, class ... Tail>
struct is_value<Tuple<Head,Tail...>> {
  static const bool value = is_value<Head>::value && is_value<Tuple<Tail...>>::value;
};

template<class... Tail>
void freeze(Tuple<Tail...>& o) {
  o.freeze();
}

template<class... Tail>
void thaw(Tuple<Tail...>& o, LazyLabel* label) {
  o.thaw(label);
}

template<class... Tail>
void finish(Tuple<Tail...>& o) {
  o.finish();
}

}
