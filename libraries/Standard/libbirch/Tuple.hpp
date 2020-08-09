/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/type.hpp"

namespace libbirch {
template<class... Args>
/**
 * Tuple.
 *
 * @tparam Args... element types.
 */
class Tuple {
  //
};

/**
 * Tuple.
 *
 * @tparam Head First element type.
 * @tparam Tail... Remaining element types.
 */
template<class Head, class ...Tail>
class Tuple<Head,Tail...> {
  template<class... Args> friend class Tuple;
public:
  Tuple() = default;

  /**
   * Constructor.
   */
  template<class... Tail1>
  Tuple(Head head, Tail1&&... tail) :
      head(head),
      tail(std::forward<Tail1>(tail)...) {
    //
  }

  /**
   * Copy constructor.
   */
  Tuple(const Tuple& o) = default;

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
   * Move constructor.
   */
  Tuple(Tuple&& o) = default;

  /**
   * Generic move constructor.
   */
  template<class Head1, class... Tail1>
  Tuple(Tuple<Head1,Tail1...>&& o) :
      head(std::move(o.head)),
      tail(std::move(o.tail)) {
    //
  }

  /**
   * Copy assignment.
   */
  Tuple& operator=(const Tuple& o) = default;

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
   * Move assignment.
   */
  Tuple& operator=(Tuple&& o) = default;

  /**
   * Generic move assignment.
   */
  template<class Head1, class... Tail1>
  Tuple& operator=(Tuple<Head1,Tail1...>&& o) {
    head = std::move(o.head);
    tail = std::move(o.tail);
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

/**
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
  Tuple(const Tuple& o) = default;

  /**
   * Generic copy constructor.
   */
  template<class Head1>
  Tuple(const Tuple<Head1>& o) :
      head(o.head) {
    //
  }

  /**s
   * Move constructor.
   */
  Tuple(Tuple&& o) = default;

  /**
   * Generic move constructor.
   */
  template<class Head1>
  Tuple(Tuple<Head1>&& o) :
      head(std::move(o.head)) {
    //
  }

  /**
   * Copy assignment.
   */
  Tuple& operator=(const Tuple& o) = default;

  /**
   * Generic copy assignment.
   */
  template<class Head1>
  Tuple& operator=(const Tuple<Head1>& o) {
    head = o.head;
    return *this;
  }

  /**
   * Move assignment.
   */
  Tuple& operator=(Tuple&& o) = default;

  /**
   * Generic move assignment.
   */
  template<class Head1>
  Tuple& operator=(Tuple<Head1>&& o) {
    head = std::move(o.head);
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

/**
 * Empty tuple.
 */
template<>
class Tuple<> {
  template<class... Args> friend class Tuple;
public:
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

template<>
struct is_value<Tuple<>> {
  static const bool value = true;
};

template<class Head, class... Tail, unsigned N>
struct is_acyclic<Tuple<Head,Tail...>,N> {
  static const bool value = is_acyclic<Head,N>::value &&
       is_acyclic<Tuple<Tail...>,N>::value;
};

template<unsigned N>
struct is_acyclic<Tuple<>,N> {
  static const bool value = true;
};

template<class... Args>
auto canonical(const Tuple<Args...>& o) {
  return o;
}

template<class... Args>
auto canonical(Tuple<Args...>&& o) {
  return std::move(o);
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
template<class Head, class... Tail>
auto make_tuple(Head&& head, Tail&&... tail) {
  return Tuple<typename std::decay<Head>::type,
      typename std::decay<Tail>::type...>(head, std::forward<Tail>(tail)...);
}

/**
 * Make a tuple with a single element.
 *
 * @tparam Head First element type.
 *
 * @param head First element.
 */
template<class Head>
auto make_tuple(Head&& head) {
  return Tuple<typename std::decay<Head>::type>(head);
}

/**
 * Make an empty tuple.
 */
inline Tuple<> make_tuple() {
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
Tuple<Head&,Tail&...> tie(Head& head, Tail&... tail) {
  return Tuple<Head&,Tail&...>(head, tail...);
}
}
