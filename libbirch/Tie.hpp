/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/type.hpp"
#include "libbirch/Tuple.hpp"

namespace libbirch {
/**
 * Tie. A Tie is a tuple of lvalue reference types. It may only be used as
 * a temporary to assign the elements of a Tuple to other variables.
 *
 * @tparam Head First element type.
 * @tparam Tail Remaining element types.
 *
 * Ties only support lvalue reference types. If there a no lvalue reference
 * types, Tuple may be used instead.
 */
template<class Head, class ...Tail>
class Tie {
  template<class Head1, class... Tail1> friend class Tie;

  static_assert(std::is_lvalue_reference<Head>::value,
      "Tie only supports lvalue reference types, try Tuple instead.");
public:
  Tie() = delete;
  Tie(const Tie&) = default;
  Tie(Tie&&) = default;
  Tie& operator=(const Tie&) = delete;
  Tie& operator=(Tie&&) = delete;

  /**
   * Constructor.
   */
  Tie(Head& head, Tail&... tail) :
      head(head),
      tail(tail...) {
    //
  }

  /**
   * Copy assignment operator.
   */
  template<IS_VALUE1(Head), IS_VALUE2(Tuple<Tail...>), class Head1,
      class... Tail1>
  Tie& operator=(const Tuple<Head1,Tail1...>& o) {
    return assign(o);
  }

  /**
   * Move assignment operator.
   */
  template<IS_VALUE1(Head),IS_VALUE2(Tie<Tail...>), class Head1,
      class... Tail1>
  Tie& operator=(Tuple<Head1,Tail1...>&& o) {
    return assign(std::move(o));
  }

  /**
   * Copy assignment.
   */
  template<IS_VALUE1(Head), IS_VALUE2(Tie<Tail...>), class Head1,
      class... Tail1>
  Tie& assign(const Tuple<Head1,Tail1...>& o) {
    head = o.head;
    tail = o.tail;
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_VALUE1(Head), IS_NOT_VALUE2(Tie<Tail...>), class Head1,
      class... Tail1>
  Tie& assign(Label* context, const Tuple<Head1,Tail1...>& o) {
    head = o.head;
    tail.assign(context, o.tail);
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_NOT_VALUE1(Head), IS_VALUE2(Tie<Tail...>), class Head1,
      class... Tail1>
  Tie& assign(Label* context, const Tuple<Head1,Tail1...>& o) {
    head.assign(context, o.head);
    tail = o.tail;
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_NOT_VALUE1(Head), IS_NOT_VALUE2(Tie<Tail...>), class Head1,
      class... Tail1>
  Tie& assign(Label* context, const Tuple<Head1,Tail1...>& o) {
    head.assign(context, o.head);
    tail.assign(context, o.tail);
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_VALUE1(Head),IS_VALUE2(Tie<Tail...>), class Head1,
      class... Tail1>
  Tie& assign(Tuple<Head1,Tail1...>&& o) {
    head = std::move(o.head);
    tail = std::move(o.tail);
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_VALUE1(Head),IS_NOT_VALUE2(Tie<Tail...>), class Head1,
      class... Tail1>
  Tie& assign(Label* context, Tuple<Head1,Tail1...>&& o) {
    head = std::move(o.head);
    tail.assign(context, std::move(o.tail));
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_NOT_VALUE1(Head),IS_VALUE2(Tie<Tail...>), class Head1,
      class... Tail1>
  Tie& assign(Label* context, Tuple<Head1,Tail1...>&& o) {
    head.assign(context, std::move(o.head));
    tail = std::move(o.tail);
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_NOT_VALUE1(Head),IS_NOT_VALUE2(Tie<Tail...>), class Head1,
      class... Tail1>
  Tie& assign(Label* context, Tuple<Head1,Tail1...>&& o) {
    head.assign(context, std::move(o.head));
    tail.assign(context, std::move(o.tail));
    return *this;
  }

private:
  /**
   * First element.
   */
  Head& head;

  /**
   * Remaining elements.
   */
  Tie<Tail&...> tail;
};

/*
 * Tie with a single element.
 *
 * @tparam Head Element type.
 */
template<class Head>
class Tie<Head> {
  template<class Head1, class... Tail1> friend class Tie;

  static_assert(std::is_lvalue_reference<Head>::value,
      "Tie only supports lvalue reference types, try Tuple instead.");
public:
  Tie() = delete;
  Tie(const Tie&) = default;
  Tie(Tie&&) = default;
  Tie& operator=(const Tie&) = delete;
  Tie& operator=(Tie&&) = delete;

  /**
   * Constructor.
   */
  Tie(Head& head) :
      head(head) {
    //
  }

  /**
   * Copy assignment operator.
   */
  template<IS_VALUE(Head), class Head1>
  Tie& operator=(const Tuple<Head1>& o) {
    return assign(o);
  }

  /**
   * Move assignment operator.
   */
  template<IS_VALUE(Head), class Head1>
  Tie& operator=(Tuple<Head1>&& o) {
    return assign(std::move(o));
  }

  /**
   * Copy assignment.
   */
  template<IS_VALUE(Head), class Head1>
  Tie& assign(const Tuple<Head1>& o) {
    head = o.head;
    return *this;
  }

  /**
   * Copy assignment.
   */
  template<IS_NOT_VALUE(Head), class Head1>
  Tie& assign(Label* context, const Tuple<Head1>& o) {
    head.assign(context, o.head);
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_VALUE(Head), class Head1>
  Tie& assign(Tuple<Head1>&& o) {
    head = std::move(o.head);
    return *this;
  }

  /**
   * Move assignment.
   */
  template<IS_NOT_VALUE(Head), class Head1>
  Tie& assign(Label* context, Tuple<Head1>&& o) {
    head.assign(context, std::move(o.head));
    return *this;
  }

private:
  /**
   * First element.
   */
  Head& head;
};

template<class Head, class ...Tail>
struct is_value<Tie<Head,Tail...>> {
  static const bool value = is_value<Head>::value && is_value<Tie<Tail...>>::value;
};
template<class Head>
struct is_value<Tie<Head>> {
  static const bool value = is_value<Head>::value;
};

}
