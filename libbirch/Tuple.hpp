/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
/**
 * Tuple.
 *
 * @ingroup libbirch
 */
template<class Head, class Enable = void, class... Tail>
class Tuple {
  //
};

/**
 * Tuple with a value head type.
 *
 * @tparam Head Head type.
 * @tparam Tail Tail types.
 */
template<class Head, class ...Tail>
class Tuple<Head,IS_VALUE(Head),Tail...> {
  template<class Head1, class Enable1, class... Tail1> friend class Tuple;
public:
  Tuple() = default;
  Tuple(const Tuple&) = default;
  Tuple(Tuple&&) = default;
  Tuple& operator=(const Tuple&) = delete;
  Tuple& operator=(Tuple&&) = delete;

  /**
   * Constructor.
   */
  Tuple(Head& head, Tail&... tail) :
      head(std::forward(head)),
      tail(std::forward(tail...)) {
    //
  }

  /**
   * Constructor.
   */
  Tuple(Label* context, Head& head, Tail& ... tail) :
      head(std::forward(head)),
      tail(context, std::forward(tail...)) {
    //
  }

  /**
   * Copy constructor.
   */
  Tuple(Label* context, const Tuple& o) :
      head(o.head),
      tail(context, o.tail) {
    //
  }

  /**
   * Move constructor.
   */
  Tuple(Label* context, Tuple&& o) :
      head(std::move(o.head)),
      tail(context, std::move(o.tail)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(o.head),
      tail(context, label, o.tail) {
    //
  }

  /**
   * Copy assignment.
   */
  template<class Head1, class ... Tail1>
  Tuple& assign(Label* context, const Tuple<Head1,Tail1...>& o) {
    head = o.head;
    tail.assign(context, o.tail);
    return *this;
  }

  /**
   * Move assignment.
   */
  template<class Head1, class ... Tail1>
  Tuple& assign(Label* context, Tuple<Head1,Tail1...>&& o) {
    head = std::move(o.head);
    tail.assign(context, std::move(o.tail));
    return *this;
  }

  void freeze() {
    tail.freeze();
  }

  void thaw(Label* label) {
    tail.thaw(label);
  }

  void finish() {
    tail.finish();
  }

private:
  /**
   * First element.
   */
  Head head;

  /**
   * Remaining elements.
   */
  Tuple<Tail...> tail;
};

/**
 * Tuple with a non-value head type.
 *
 * @tparam Head Head type.
 * @tparam Tail Tail types.
 */
template<class Head, class ...Tail>
class Tuple<Head,IS_NOT_VALUE(Head),Tail...> {
  template<class Head1, class Enable1, class... Tail1> friend class Tuple;
public:
  Tuple() = default;
  Tuple(const Tuple&) = default;
  Tuple(Tuple&&) = default;
  Tuple& operator=(const Tuple&) = delete;
  Tuple& operator=(Tuple&&) = delete;

  /**
   * Constructor.
   */
  Tuple(Label* context, Head& head, Tail&... tail) :
      head(context, std::forward(head)),
      tail(context, std::forward(tail...)) {
    //
  }

  /**
   * Copy constructor.
   */
  Tuple(Label* context, const Tuple& o) :
      head(context, o.head),
      tail(context, o.tail) {
    //
  }

  /**
   * Move constructor.
   */
  Tuple(Label* context, Tuple&& o) :
      head(context, std::move(o.head)),
      tail(context, std::move(o.tail)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(context, label, o.head),
      tail(context, label, o.tail) {
    //
  }

  /**
   * Copy assignment.
   */
  template<class Head1, class ... Tail1>
  Tuple& assign(Label* context, const Tuple<Head1,Tail1...>& o) {
    head.assign(context, o.head);
    tail.assign(context, o.tail);
    return *this;
  }

  /**
   * Move assignment.
   */
  template<class Head1, class ... Tail1>
  Tuple& assign(Label* context, Tuple<Head1,Tail1...>&& o) {
    head.assign(context, std::move(o.head));
    tail.assign(context, std::move(o.tail));
    return *this;
  }

  void freeze() {
    head.freeze();
    tail.freeze();
  }

  void thaw(Label* label) {
    head.thaw(label);
    tail.thaw(label);
  }

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
   * Remaining elements.
   */
  Tuple<Tail...> tail;
};

/**
 * Tuple with a value head type, and no tail.
 *
 * @param Head Value type.
 */
template<class Head>
class Tuple<Head,IS_VALUE(Head)> {
  template<class Head1, class Enable1, class... Tail1> friend class Tuple;
public:
  Tuple() = default;
  Tuple(const Tuple&) = default;
  Tuple(Tuple&&) = default;
  Tuple& operator=(const Tuple&) = default;
  Tuple& operator=(Tuple&&) = default;

  /**
   * Constructor.
   */
  Tuple(Head& head) :
      head(std::forward(head)) {
    //
  }

  /**
   * Constructor.
   */
  Tuple(Label* context, Head& head) :
      head(std::forward(head)) {
    //
  }

  /**
   * Copy constructor.
   */
  Tuple(Label* context, const Tuple& o) :
      head(o.head) {
    //
  }

  /**
   * Move constructor.
   */
  Tuple(Label* context, Tuple&& o) :
      head(std::move(o.head)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(o.head) {
    //
  }

  /**
   * Copy assignment.
   */
  template<class Head1>
  Tuple& assign(Label* context, const Tuple<Head1>& o) {
    head = o.head;
    return *this;
  }

  /**
   * Move assignment.
   */
  template<class Head1>
  Tuple& assign(Label* context, Tuple<Head1>&& o) {
    head = std::move(o.head);
    return *this;
  }

  void freeze() {
    //
  }

  void thaw(Label* label) {
    //
  }

  void finish() {
    //
  }

private:
  /**
   * First element.
   */
  Head head;
};

/**
 * Tuple with a non-value head type, and no tail.
 *
 * @param Head Value type.
 */
template<class Head>
class Tuple<Head,IS_NOT_VALUE(Head)> {
  template<class Head1, class Enable1, class... Tail1> friend class Tuple;
public:
  Tuple() = default;
  Tuple(const Tuple&) = default;
  Tuple(Tuple&&) = default;
  Tuple& operator=(const Tuple&) = delete;
  Tuple& operator=(Tuple&&) = delete;

  /**
   * Constructor.
   */
  Tuple(Label* context, Head& head) :
      head(context, std::forward(head)) {
    //
  }

  /**
   * Copy constructor.
   */
  Tuple(Label* context, const Tuple& o) :
      head(context, o.head) {
    //
  }

  /**
   * Move constructor.
   */
  Tuple(Label* context, Tuple&& o) :
      head(context, std::move(o.head)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  Tuple(Label* context, Label* label, const Tuple& o) :
      head(context, label, o.head) {
    //
  }

  /**
   * Copy assignment.
   */
  template<class Head1>
  Tuple& assign(Label* context, const Tuple<Head1>& o) {
    head.assign(context, o.head);
    return *this;
  }

  /**
   * Move assignment.
   */
  template<class Head1>
  Tuple& assign(Label* context, Tuple<Head1>&& o) {
    head.assign(context, std::move(o.head));
    return *this;
  }

  void freeze() {
    head.freeze();
  }

  void thaw(Label* label) {
    head.thaw(label);
  }

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
