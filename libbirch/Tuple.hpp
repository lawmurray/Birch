/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
template<class... Args>
class Tuple {
  //
};

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
class Tuple<Head,Tail...> {
  template<class... Args> friend class Tuple;
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
  template<class... Args> friend class Tuple;
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

/*
 * Empty tuple.
 */
template<>
class Tuple<> {
  template<class... Args> friend class Tuple;
public:
  Tuple() = default;

  /**
   * Accept visitor.
   */
  template<class Visitor>
  void accept_(const Visitor& v) {
    //
  }
};

template<class Head, class... Tail>
struct is_value<Tuple<Head,Tail...>> {
  static const bool value = is_value<Head>::value &&
      is_value<Tuple<Tail...>>::value;
};

template<class Head>
struct is_value<Tuple<Head>> {
  static const bool value = is_value<Head>::value;
};

template<>
struct is_value<Tuple<>> {
  static const bool value = true;
};

template<class Head, class... Tail, unsigned N>
struct is_acyclic<Tuple<Head,Tail...>,N> {
  static const bool value = is_acyclic<Head,N>::value &&
       is_acyclic<Tuple<Tail...>,N>::value;
};

template<class Head, unsigned N>
struct is_acyclic<Tuple<Head>,N> {
  static const bool value = is_acyclic<Head,N>::value;
};

template<unsigned N>
struct is_acyclic<Tuple<>,N> {
  static const bool value = true;
};

template<class Head, class Tail>
auto canonical(const Tuple<Head,Tail>& o) {
  return o;
}

/**
 * Make a tuple.
 *
 * @tparam Head First element type.
 * @tparam Tail Remaining element types.
 *
 * @param head First element.
 * @param tail Remaining elements.
 */
template<class Head, class ... Tail>
auto make_tuple(const Head& head, const Tail&... tail) {
  return Tuple<Head,Tail...>(head, tail...);
}

/**
 * Make a tuple with a single element.
 *
 * @tparam Head First element type.
 *
 * @param head First element.
 */
template<class Head>
auto make_tuple(const Head& head) {
  return Tuple<Head>(head);
}

/**
 * Make an empty tuple.
 */
inline auto make_tuple() {
  return Tuple<>();
}

/**
 * Tie a tuple.
 *
 * @tparam Head First element type.
 * @tparam Tail Remaining element types.
 *
 * @param head First element.
 * @param tail Remaining elements.
 */
template<class Head, class ... Tail>
auto tie(Head&& head, Tail&&... tail) {
  return Tuple<Head&,Tail&...>(head, tail...);
}

/**
 * Tie a constant tuple.
 *
 * @tparam Head First element type.
 * @tparam Tail Remaining element types.
 *
 * @param head First element.
 * @param tail Remaining elements.
 */
template<class Head, class ... Tail>
auto const_tie(const Head& head, const Tail&... tail) {
  return Tuple<const Head&,const Tail&...>(head, tail...);
}
}
