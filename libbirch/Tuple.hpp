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
public:
  Tuple() = default;

  /**
   * Constructor.
   */
  Tuple(Head head, Tail... tail) :
      head(head),
      tail(tail...) {
    //
  }

  /**
   * Copy constructor.
   */
  Tuple(const Tuple& o) :
      head(o.head),
      tail(o.tail) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class Head1, class... Tail1>
  Tuple(const Tuple<Head1,Tail1...>& o) :
      head(o.head),
      tail(o.tail) {
    //
  }

  /**
   * Copy assignment.
   */
  Tuple& operator=(const Tuple& o) {
    head = o.head;
    tail = o.tail;
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class Head1, class... Tail1>
  Tuple& operator=(const Tuple<Head1,Tail1...>& o) {
    head = o.head;
    tail = o.tail;
    return *this;
  }

  /**
   * Accept visitor.
   */
  template<class Visitor>
  void accept_(const Visitor& v) {
    v.visit(head);
    v.visit(tail);
  }

  /**
   * Get element.
   */
  template<int n, std::enable_if_t<n == 0,int> = 0>
  auto& get() {
    return head;
  }

  /**
   * Get element.
   */
  template<int n, std::enable_if_t<n == 0,int> = 0>
  const auto& get() const {
    return head;
  }

  /**
   * Get element.
   */
  template<int n, std::enable_if_t<(n > 0),int> = 0>
  auto& get() {
    return tail.template get<n - 1>();
  }

  /**
   * Get element.
   */
  template<int n, std::enable_if_t<(n > 0),int> = 0>
  const auto& get() const {
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
public:
  Tuple() = default;

  /**
   * Constructor.
   */
  Tuple(Head head) :
      head(head) {
    //
  }

  /**
   * Copy constructor.
   */
  Tuple(const Tuple& o) :
      head(o.head) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class Head1>
  Tuple(const Tuple<Head1>& o) :
      head(o.head) {
    //
  }

  /**
   * Copy assignment.
   */
  Tuple& operator=(const Tuple& o) {
    head = o.head;
    return *this;
  }

  /**
   * Generic copy assignment.
   */
  template<class Head1>
  Tuple& operator=(const Tuple<Head1>& o) {
    head = o.head;
    return *this;
  }

  /**
   * Accept visitor.
   */
  template<class Visitor>
  void accept_(const Visitor& v) {
    v.visit(head);
  }

  /**
   * Get element.
   */
  template<int n, std::enable_if_t<n == 0,int> = 0>
  auto& get() {
    return head;
  }

  /**
   * Get element.
   */
  template<int n, std::enable_if_t<n == 0,int> = 0>
  const auto& get() const {
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
