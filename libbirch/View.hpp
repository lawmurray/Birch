/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"
#include "libbirch/Index.hpp"
#include "libbirch/Range.hpp"

#include <cstddef>

namespace bi {
/**
 * Empty view.
 *
 * @ingroup libbirch
 *
 * @see View
 */
struct EmptyView {
  template<class T1>
  void offsets(T1* out) const {
    //
  }

  template<class T1>
  void lengths(T1* out) const {
    //
  }

  static constexpr int count() {
    return 0;
  }

  static constexpr int rangeCount() {
    return 0;
  }

  size_t size() const {
    return 1;
  }

  bool operator==(const EmptyView& o) const {
    return true;
  }

  template<class View1>
  bool operator==(const View1& o) const {
    return false;
  }

  template<class View1>
  bool operator!=(const View1& o) const {
    return !(*this == o);
  }
};

/**
 * Nonempty view.
 *
 * @ingroup libbirch
 *
 * @tparam Head Range or Index type.
 * @tparam Tail View type.
 *
 * A view describes the active elements over `D` dimensions of an array. It
 * consists of a @em head range or index for the first dimension, and a
 * @em tail view for the remaining `D - 1` dimensions, recursively. The tail
 * view is EmptyView for the last dimension.
 */
template<class Head, class Tail>
struct NonemptyView {
  /**
   * Head type.
   */
  typedef Head head_type;

  /**
   * Tail type.
   */
  typedef Tail tail_type;

  /**
   * Default constructor.
   */
  NonemptyView() {
    //
  }

  /**
   * Generic constructor.
   */
  template<class Head1, class Tail1>
  NonemptyView(const Head1 head, const Tail1 tail) :
      head(head),
      tail(tail) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class Head1, class Tail1>
  NonemptyView(const NonemptyView<Head1,Tail1>& o) :
      head(o.head),
      tail(o.tail) {
    //
  }

  /**
   * @name Collections
   */
  //@{
  /**
   * Get offsets.
   *
   * @tparam T1 Integer type.
   *
   * @param[out] out Array assumed to have at least count() elements.
   */
  template<class T1>
  void offsets(T1* out) const {
    *out = head.offset;
    tail.offsets(out + 1);
  }

  /**
   * Get lengths.
   *
   * @tparam T1 Integer type.
   *
   * @param[out] out Array assumed to have at least count() elements.
   */
  template<class T1>
  void lengths(T1* out) const {
    *out = head.length;
    tail.lengths(out + 1);
  }
  //@}

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
  size_t size() const {
    return head.length * tail.size();
  }
  //@}

  /**
   * @name Indexing and iterating
   */
  //@{
  /**
   * Equality operator.
   */
  template<class Head1, class Tail1>
  bool operator==(const NonemptyView<Head1,Tail1>& o) const {
    return head == o.head && tail == o.tail;
  }

  /**
   * Equality operator.
   */
  bool operator==(const EmptyView& o) const {
    return false;
  }

  /**
   * Unequality operator.
   */
  template<class View1>
  bool operator!=(const View1& o) const {
    return !(*this == o);
  }

  /**
   * Head.
   */
  Head head;

  /**
   * Tail.
   */
  Tail tail;
};
}
