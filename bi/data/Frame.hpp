/**
 * @file
 */
#pragma once

#include "bi/data/constant.hpp"
#include "bi/data/Span.hpp"
#include "bi/data/Index.hpp"
#include "bi/data/Range.hpp"
#include "bi/data/View.hpp"

#include <cstddef>

namespace bi {
/**
 * Empty frame.
 *
 * @ingroup library
 *
 * @see NonemptyFrame
 */
struct EmptyFrame {
  EmptyFrame() {
    //
  }

  EmptyFrame operator()(const EmptyView& o) const {
    return EmptyFrame();
  }

  bool conforms(const EmptyFrame& o) const {
    return true;
  }

  template<class Frame1>
  bool conforms(const Frame1& o) const {
    return false;
  }

  template<class Frame1>
  Frame1 prepend(const Frame1& o) const {
    return o;
  }

  void coord(const int_t serial, EmptyView& o) {
    //
  }

  int_t length(const int i) const {
    assert(i == 0);
    return 1;
  }

  int_t stride(const int i) const {
    assert(i == 0);
    return 1;
  }

  int_t lead(const int i) const {
    assert(i == 0);
    return 1;
  }

  template<class T1>
  void lengths(T1* out) const {
    //
  }

  template<class T1>
  void strides(T1* out) const {
    //
  }

  template<class T1>
  void leads(T1* out) const {
    //
  }

  static constexpr int count() {
    return 0;
  }

  static constexpr int_t size() {
    return 1;
  }

  static constexpr int_t volume() {
    return 1;
  }

  template<class View>
  int_t serial(const View& o) const {
    return 0;
  }

  bool contiguous() const {
    return true;
  }

  template<bool is_not_contiguous_value = false, bool is_contiguous_value =
      true>
  EmptyView block(bool contiguous = true) const {
    return EmptyView();
  }

  bool operator==(const EmptyFrame& o) const {
    return true;
  }

  template<class Frame1>
  bool operator==(const Frame1& o) const {
    return false;
  }

  template<class Frame1>
  bool operator!=(const Frame1& o) const {
    return !operator==(o);
  }

  EmptyFrame& operator*=(const int_t n) {
    return *this;
  }

  EmptyFrame operator*(const int_t n) const {
    EmptyFrame result(*this);
    result *= n;
    return result;
  }
};

/**
 * Nonempty frame.
 *
 * @ingroup library
 *
 * @tparam Tail Frame type.
 * @tparam Head Span type.
 *
 * A frame describes the `D` dimensions of an array. It consists of a
 * @em tail frame describing the first `D - 1` dimensions, recursively, and a
 * @em head Span describing the remaining dimension. The tail frame is
 * EmptyFrame for the first dimension.
 */
template<class Tail, class Head>
struct NonemptyFrame {
  /**
   * Tail type.
   */
  typedef Tail tail_type;

  /**
   * Head type.
   */
  typedef Head head_type;

  /**
   * Generic constructor.
   */
  template<class Tail1, class Head1>
  NonemptyFrame(const Tail1 tail, const Head1 head) :
      tail(tail),
      head(head) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<class Tail1, class Head1>
  NonemptyFrame(const NonemptyFrame<Tail1,Head1>& o) :
      tail(o.tail),
      head(o.head) {
    //
  }

  /**
   * View operator.
   */
  template<class Tail1, int_t other_offset_value, int_t other_length_value,
      int_t other_stride_value>
  auto operator()(
      const NonemptyView<Tail1,
          Range<other_offset_value,other_length_value,other_stride_value>>& o) const {
    return NonemptyFrame<decltype(tail(o.tail)),decltype(head(o.head))>(
        tail(o.tail), head(o.head));
  }

  /**
   * View operator.
   */
  template<class Tail1, int_t other_offset_value>
  auto operator()(
      const NonemptyView<Tail1,Index<other_offset_value>>& o) const {
    return tail(o.tail) * head.lead;
  }

  /**
   * Does this frame conform to another? Two frames conform if their spans
   * conform.
   */
  bool conforms(const EmptyFrame& o) const {
    return false;
  }

  /**
   * Does this frame conform to another? Two frames conform if their spans
   * conform.
   */
  template<class Frame1>
  bool conforms(const Frame1& o) const {
    return tail.conforms(o.tail)/* && head.conforms(o.head)*/;
  }

  /**
   * Append dimensions from another frame to the right of this one.
   */
  template<class Frame1>
  auto prepend(const Frame1& o) const {
    auto tail = this->tail.prepend(o);
    return NonemptyFrame<decltype(tail),Head>(tail, head);
  }

  /**
   * Expand serial index into coordinates.
   *
   * @param serial The serial index.
   * @param[out] view The view in which to write the coordinates.
   */
  template<class View>
  void coord(const int_t serial, View& view) {
    tail.coord(serial / head.length, view.tail);
    view.head.offset = serial % head.length;
  }

  /**
   * @name Queries
   */
  //@{
  /**
   * Get the length of the @p i th dimension.
   */
  int_t length(const int i) const {
    if (i == count() - 1) {
      return head.length;
    } else {
      return tail.length(i - 1);
    }
  }

  /**
   * Get the stride of the @p i th dimension.
   */
  int_t stride(const int i) const {
    if (i == count() - 1) {
      return head.stride;
    } else {
      return tail.stride(i - 1);
    }
  }

  /**
   * Get the lead of the @p i th dimension.
   */
  int_t lead(const int i) const {
    if (i == count() - 1) {
      return head.lead;
    } else {
      return tail.lead(i - 1);
    }
  }
  //@}

  /**
   * @name Collections
   */
  //@{
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

  /**
   * Get leads.
   *
   * @tparam T1 Integer type.
   *
   * @param[out] out Array assumed to have at least count() elements.
   */
  template<class T1>
  void leads(T1* out) const {
    tail.leads(out);
    *(out + Tail::count()) = head.lead;
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
   * Product of all lengths.
   */
  int_t size() const {
    return tail.size()*head.length;
  }

  /**
   * Product of all leads.
   */
  int_t volume() const {
    return tail.volume()*head.lead;
  }

  /**
   * Serial offset for a view.
   */
  template<class View>
  int_t serial(const View& o) const {
    return tail.serial(o.tail) * head.lead + head.stride * o.head.offset;
  }

  /**
   * Are all elements stored contiguously in memory?
   */
  bool contiguous() const {
    return this->length == 1 || (count() == 1 && head.stride == 1)
        || (tail.contiguous() && this->lead == this->length);
  }
  //@}

  /**
   * @name Iterating
   */
  //@{
  /**
   * View describing contiguous blocks in storage.
   *
   * @tparam is_not_contiguous Is the frame statically known to be not
   * contiguous at this point?
   * @tparam is_contiguous Is the frame statically known to be contiguous
   * at this point?
   */
  template<bool is_not_contiguous_value = false, bool is_contiguous_value =
      true>
  auto block(bool contiguous = true) const {
    /* tail */
    static const bool new_is_not_contiguous_value = is_not_contiguous_value
        || Head::stride_value > 1
        || (Head::length_value != Head::lead_value
            && Head::length_value * Head::lead_value > 0);
    static const bool new_is_contiguous_value = is_contiguous_value
        && Head::length_value == Head::lead_value
        && Head::length_value * Head::lead_value > 0;

    bool new_contiguous = contiguous && head.length == head.lead;

    auto new_tail = tail.template block<new_is_not_contiguous_value,
        new_is_contiguous_value>(new_contiguous);

    /* head */
    static const int_t new_offset_value = 0;
    static const int_t new_length_value =
        is_not_contiguous_value ?
            1 :
            ((is_contiguous_value && Head::stride_value == 1) ?
                Head::length_value : mutable_value);
    static const int_t new_stride_value = 1;

    int_t new_offset = 0;
    int_t new_length = (head.stride == 1) ? head.length : 1;
    int_t new_stride = 1;

    auto new_head = Range<new_offset_value,new_length_value,new_stride_value>(
        new_offset, new_length, new_stride);

    /* combine */
    return NonemptyView<decltype(new_tail),decltype(new_head)>(new_tail,
        new_head);
  }

  /**
   * Equality operator.
   */
  template<class Head1, class Tail1>
  bool operator==(const NonemptyFrame<Head1,Tail1>& o) const {
    return tail == o.tail && head == o.head;
  }

  /**
   * Equality operator.
   */
  bool operator==(const EmptyFrame& o) const {
    return false;
  }

  /**
   * Unequality operator.
   */
  template<class Frame1>
  bool operator!=(const Frame1& o) const {
    return !(*this == o);
  }

  /**
   * Multiply stride of rightmost dimension.
   *
   * @todo Take index instead of n.
   */
  NonemptyFrame<Tail,Head>& operator*=(const int_t n) {
    head *= n;
    return *this;
  }

  /**
   * Multiply stride of rightmost dimension.
   *
   * @todo Take index instead of n.
   */
  auto operator*(const int_t n) const {
    auto head = this->head * n;
    return NonemptyFrame<Tail,decltype(head)>(this->tail, head);
  }

  /**
   * Tail
   */
  Tail tail;

  /**
   * Head.
   */
  Head head;
};

/**
 * Default frame for `D` dimensions.
 */
template<int D>
struct DefaultFrame {
  typedef NonemptyFrame<typename DefaultFrame<D - 1>::type,Span<>> type;
};
template<>
struct DefaultFrame<0> {
  typedef EmptyFrame type;
};
}
