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
 * @tparam Head First element type.
 * @tparam ...Tail Remaining element types.
 *
 * Tuples do not support lvalue reference types. If all types are lvalue
 * reference types, Tie may be used instead.
 */
template<class Head, class ...Tail>
class Tuple {
  template<class Head1, class... Tail1> friend class Tuple;
  template<class Head1, class... Tail1> friend class Tie;

  static_assert(!std::is_lvalue_reference<Head>::value,
      "Tuple does not support lvalue reference types, try Tie instead.");
public:
  Tuple() = default;

  /**
   * Constructor.
   */
  Tuple(const Head& head, const Tail&... tail) :
      head(head),
      tail(tail...) {
    //
  }

  /**
   * Generic constructor.
   */
  template<class Head1, class... Tail1>
  Tuple(const Tuple<Head1,Tail1...>& o) :
      head(o.head),
      tail(o.tail) {
    //
  }

  /**
   * Accept function.
   */
  template<class Visitor>
  void accept(Visitor& v) {
    v.visit(head, tail);
  }

  /**
   * Get element.
   */
  template<int n, std::enable_if_t<n == 0,int> = 0>
  auto get() const {
    return head;
  }

  /**
   * Get element.
   */
  template<int n, std::enable_if_t<(n > 0),int> = 0>
  auto get() const {
    return tail.template get<n - 1>();
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

/*
 * Tuple with a single element.
 *
 * @tparam Head Element type.
 */
template<class Head>
class Tuple<Head> {
  template<class Head1, class... Tail1> friend class Tuple;
  template<class Head1, class... Tail1> friend class Tie;

  static_assert(!std::is_lvalue_reference<Head>::value,
      "Tuple does not support lvalue reference types, try Tie instead.");
public:
  Tuple() = default;

  /**
   * Constructor.
   */
  Tuple(const Head& head) :
      head(head) {
    //
  }

  /**
   * Generic constructor.
   */
  template<class Head1>
  Tuple(const Tuple<Head1>& o) :
      head(o.head) {
    //
  }

  /**
   * Accept function.
   */
  template<class Visitor>
  void accept(Visitor& v) {
    v.visit(head);
  }

  /**
   * Get element.
   */
  template<int n, std::enable_if_t<n == 0,int> = 0>
  auto get() const {
    return head;
  }

private:
  /**
   * First element.
   */
  Head head;
};

template<class Head, class ...Tail>
struct is_value<Tuple<Head,Tail...>> {
  static const bool value = is_value<Head>::value && is_value<Tuple<Tail...>>::value;
};
template<class Head>
struct is_value<Tuple<Head>> {
  static const bool value = is_value<Head>::value;
};
}
