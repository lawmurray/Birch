/**
 * @file
 */
#pragma once

#include "bi/data/constant.hpp"
#include "bi/data/Index.hpp"
#include "bi/data/Range.hpp"

#include <cstddef>

namespace bi {
/**
 * Empty view.
 *
 * @ingroup library
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

  template<class T1>
  void strides(T1* out) const {
    //
  }

  static constexpr int count() {
    return 0;
  }

  int_t size() const {
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
 * @ingroup library
 *
 * @tparam Tail View type.
 * @tparam Head Range or Index type.
 *
 * A view describes the active elements over `D` dimensions of an array. It
 * consists of a @em tail view for the first `D - 1` dimensions, recursively,
 * and a @em head Range or Index for the remaining dimension. The tail view
 * is EmptyView for the first dimension.
 */
template<class Tail, class Head>
struct NonemptyView {
  /**
   * Tail type.
   */
  typedef Tail tail_type;

  /**
   * Head type.
   */
  typedef Head head_type;

  /**
   * Default constructor.
   */
  NonemptyView() {
    //
  }

  /**
   * Generic constructor.
   */
  template<class Tail1, class Head1>
  NonemptyView(const Tail1 tail, const Head1 head) :
      tail(tail),
      head(head) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class Tail1, class Head1>
  NonemptyView(const NonemptyView<Tail1,Head1>& o) :
      tail(o.tail),
      head(o.head) {
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
    tail.offsets(out);
    *(out + Tail::count()) = head.offset;
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
    tail.lengths(out);
    *(out + Tail::count()) = head.length;
  }

  /**
   * Get strides.
   *
   * @tparam T1 Integer type.
   *
   * @param[out] out Array assumed to have at least count() elements.
   */
  template<class T1>
  void strides(T1* out) const {
    tail.strides(out);
    *(out + Tail::count()) = head.stride;
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
    return Tail::count() + 1;
  }

  /**
   * Size (the product of all lengths).
   */
  int_t size() const {
    return tail.size() * head.length;
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
    return tail == o.tail && head == o.head;
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
   * Tail.
   */
  Tail tail;

  /**
   * Head.
   */
  Head head;
};

/**
 * Default view for `D`-dimensional indexing of a single element.
 */
template<int D>
struct DefaultView {
  typedef NonemptyView<typename DefaultView<D - 1>::type,Index<>> type;
};
template<>
struct DefaultView<0> {
  typedef EmptyView type;
};
}
