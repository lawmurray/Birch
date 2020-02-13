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
  /**
   * Constructor.
   */
  Tie(Head& head, Tail&... tail) :
      head(head),
      tail(tail...) {
    //
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
  /**
   * Constructor.
   */
  Tie(Head& head) :
      head(head) {
    //
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
