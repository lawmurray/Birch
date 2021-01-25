/**
 * @file
 */
#pragma once

#include "libbirch/Index.hpp"
#include "libbirch/Range.hpp"

namespace libbirch {
/**
 * Empty slice.
 *
 * @ingroup libbirch
 *
 * @see Slice
 */
struct EmptySlice {
  static constexpr int count() {
    return 0;
  }

  static constexpr int rangeCount() {
    return 0;
  }

  int64_t size() const {
    return 1;
  }
};

/**
 * Slice into an array
 *
 * @ingroup libbirch
 *
 * @tparam Head Range or Index type.
 * @tparam Tail Slice type.
 *
 * A slice describes the active elements over `D` dimensions of an array. It
 * consists of a @em head Range or Index for the first dimension, and a
 * @em tail slice for the remaining `D - 1` dimensions, recursively. The tail
 * slice is EmptySlice for the last dimension.
 */
template<class Head, class Tail>
struct Slice {
  /**
   * Constructor.
   */
  Slice() = default;

  /**
   * Generic constructor.
   */
  template<class Head1, class Tail1>
  Slice(const Head1 head, const Tail1 tail) :
      head(head),
      tail(tail) {
    //
  }

  /**
   * Copy constructor.
   */
  Slice(const Slice<Head,Tail>& o) = default;

  /**
   * Generic copy constructor.
   */
  template<class Head1, class Tail1>
  Slice(const Slice<Head1,Tail1>& o) :
      head(o.head),
      tail(o.tail) {
    //
  }

  /**
   * @name Reductions
   */
  //@{
  /**
   * Number of dimensions.
   */
  static constexpr int count() {
    return 1 + Tail::count();
  }

  /**
   * Number of ranges.
   */
  static constexpr int rangeCount() {
    return Head::rangeCount() + Tail::rangeCount();
  }

  /**
   * Size (the product of all lengths).
   */
  int64_t size() const {
    return head.length * tail.size();
  }
  //@}

  /**
   * Head.
   */
  Head head;

  /**
   * Tail.
   */
  Tail tail;
};

/**
 * Default slice for `D`-dimensional indexing of a single element.
 */
template<int D>
struct DefaultSlice {
  typedef Slice<Index<>,typename DefaultSlice<D - 1>::type> type;
};
template<>
struct DefaultSlice<0> {
  typedef EmptySlice type;
};

/**
 * Make a slice, no arguments.
 *
 * @ingroup libbirch
 */
inline auto make_slice() {
  return EmptySlice();
}

/**
 * Make a slice, single argument.
 *
 * @ingroup libbirch
 */
template<int64_t offset_value, int64_t length_value>
auto make_slice(const Range<offset_value,length_value>& arg) {
  auto head = arg;
  auto tail = make_slice();
  return Slice<decltype(head),decltype(tail)>(head, tail);
}

/**
 * Make a slice, single argument.
 *
 * @ingroup libbirch
 */
inline auto make_slice(const int64_t arg) {
  auto head = make_index(arg);
  auto tail = make_slice();
  return Slice<decltype(head),decltype(tail)>(head, tail);
}

/**
 * Make a slice, multiple arguments.
 *
 * @ingroup libbirch
 */
template<int64_t offset_value, int64_t length_value, class ... Args>
auto make_slice(const Range<offset_value,length_value>& arg, Args&&... args) {
  auto head = arg;
  auto tail = make_slice(std::forward<Args>(args)...);
  return Slice<decltype(head),decltype(tail)>(head, tail);
}

/**
 * Make a slice, multiple arguments.
 *
 * @ingroup libbirch
 */
template<class ... Args>
auto make_slice(const int64_t arg, Args&&... args) {
  auto head = make_index(arg);
  auto tail = make_slice(std::forward<Args>(args)...);
  return Slice<decltype(head),decltype(tail)>(head, tail);
}

}
