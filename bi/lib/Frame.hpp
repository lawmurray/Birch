/**
 * @file
 */
#pragma once

#include "bi/lib/global.hpp"
#include "bi/lib/Span.hpp"
#include "bi/lib/Index.hpp"
#include "bi/lib/Range.hpp"
#include "bi/lib/View.hpp"
#include "bi/lib/Eigen.hpp"

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

  /**
   * Special constructor for Eigen integration where all matrices and vectors
   * are treated as matrices, with row and column counts. If this constructor,
   * is reached, it means the array is a vector, and @p length should be one
   * (and is checked for this).
   */
  EmptyFrame(const Eigen::Index cols) {
    assert(cols == 1);
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

  bool conforms(const Eigen::Index rows, const Eigen::Index cols) {
    return false;
  }

  bool conforms(const Eigen::Index rows) {
    return false;
  }

  void resize(const EmptyFrame& o) {
    //
  }

  void resize(const Eigen::Index rows) {
    //
  }

  template<class Frame1>
  Frame1 prepend(const Frame1& o) const {
    return o;
  }

  ptrdiff_t offset(const ptrdiff_t n) {
    return 0;
  }

  size_t length(const int i) const {
    assert(false);
  }

  ptrdiff_t stride(const int i) const {
    assert(false);
  }

  size_t lead(const int i) const {
    assert(false);
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

  void setLength(const int i, const size_t value) {
    assert(false);
  }

  void setStride(const int i, const ptrdiff_t value) {
    assert(false);
  }

  void setLead(const int i, const size_t value) {
    assert(false);
  }

  template<class T1>
  void setLengths(T1* in) {
    //
  }

  template<class T1>
  void setStrides(T1* in) {
    //
  }

  template<class T1>
  void setLeads(T1* in) {
    //
  }

  static constexpr int count() {
    return 0;
  }

  static constexpr size_t size() {
    return 1;
  }

  static constexpr size_t volume() {
    return 1;
  }

  static constexpr size_t block() {
    return 1;
  }

  template<class View>
  ptrdiff_t serial(const View& o) const {
    return 0;
  }

  bool contiguous() const {
    return true;
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

  EmptyFrame& operator*=(const ptrdiff_t n) {
    return *this;
  }

  EmptyFrame operator*(const ptrdiff_t n) const {
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
   * Default constructor (for zero-size frame).
   */
  NonemptyFrame() {
    //
  }

  /*
   * Special constructors for Eigen integration where all matrices and vectors
   * are treated as matrices, with row and column counts.
   */
  NonemptyFrame(const Eigen::Index rows, const Eigen::Index cols) :
      tail(tail_type::count() == 0 ? cols : rows),
      head(tail_type::count() == 0 ? rows : cols) {
    //
  }
  NonemptyFrame(const Eigen::Index rows) :
      head(rows) {
    assert(tail_type::count() == 0);
  }

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
  template<class Tail1, ptrdiff_t other_offset_value,
      size_t other_length_value, ptrdiff_t other_stride_value>
  auto operator()(
      const NonemptyView<Tail1,
          Range<other_offset_value,other_length_value,other_stride_value>>& o) const {
    /* pre-conditions */
    assert(o.head.offset >= 0);
    assert(o.head.length == 0 || (o.head.offset + (o.head.length - 1) * o.head.stride < head.length));

    return NonemptyFrame<decltype(tail(o.tail)),decltype(head(o.head))>(
        tail(o.tail), head(o.head));
  }

  /**
   * View operator.
   */
  template<class Tail1, ptrdiff_t other_offset_value>
  auto operator()(
      const NonemptyView<Tail1,Index<other_offset_value>>& o) const {
    /* pre-condition */
    assert(o.head.offset >= 0 && o.head.offset < head.length);

    return tail(o.tail) * head.lead;
  }

  /**
   * Does this frame conform to another? Two frames conform if their spans
   * conform.
   */
  bool conforms(const EmptyFrame& o) const {
    return false;
  }
  template<class Frame1>
  bool conforms(const Frame1& o) const {
    return tail.conforms(o.tail) && head.conforms(o.head);
  }
  bool conforms(const Eigen::Index rows, const Eigen::Index cols) {
    return tail.conforms(tail_type::count() == 0 ? cols : rows) &&
        head.conforms(tail_type::count() == 0 ? rows : cols);
  }
  bool conforms(const Eigen::Index rows) {
    return head.conforms(rows);
  }

  /**
   * Resize this frame to conform to another.
   */
  template<class Frame1>
  void resize(const Frame1& o) {
    tail.resize(o.tail);
    head.resize(o.head);
  }
  void resize(const Eigen::Index rows, const Eigen::Index cols) {
    tail.resize(tail_type::count() == 0 ? cols : rows);
    head.resize(tail_type::count() == 0 ? rows : cols);
  }
  void resize(const Eigen::Index rows) {
    head.resize(rows);
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
   * Compute the offset to the @p n th element in storage order.
   *
   * @param n Element number.
   */
  ptrdiff_t offset(const ptrdiff_t n) {
    ptrdiff_t q = n / head.length;
    ptrdiff_t r = n % head.length;

    return tail.offset(q) * head.lead + r * head.stride;
  }

  /**
   * @name Getters
   */
  //@{
  /**
   * Get the length of the @p i th dimension.
   */
  size_t length(const int i) const {
    /* pre-condition */
    assert(i >= 0 && i < count());
    if (i == count() - 1) {
      return head.length;
    } else {
      return tail.length(i);
    }
  }

  /**
   * Get the stride of the @p i th dimension.
   */
  ptrdiff_t stride(const int i) const {
    /* pre-condition */
    assert(i >= 0 && i < count());

    if (i == count() - 1) {
      return head.stride;
    } else {
      return tail.stride(i);
    }
  }

  /**
   * Get the lead of the @p i th dimension.
   */
  size_t lead(const int i) const {
    /* pre-condition */
    assert(i >= 0 && i < count());

    if (i == count() - 1) {
      return head.lead;
    } else {
      return tail.lead(i);
    }
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
   * @name Setters
   */
  //@{
  /**
   * Set the length of the @p i th dimension.
   */
  void setLength(const int i, const size_t value) {
    /* pre-condition */
    assert(i >= 0 && i < count());
    if (i == count() - 1) {
      head.length = value;
    } else {
      tail.setLength(i, value);
    }
  }

  /**
   * Set the stride of the @p i th dimension.
   */
  void setStride(const int i, const ptrdiff_t value) {
    /* pre-condition */
    assert(i >= 0 && i < count());

    if (i == count() - 1) {
      head.stride = value;
    } else {
      tail.setStride(i, value);
    }
  }

  /**
   * Set the lead of the @p i th dimension.
   */
  void setLead(const int i, const size_t value) {
    /* pre-condition */
    assert(i >= 0 && i < count());

    if (i == count() - 1) {
      head.lead = value;
    } else {
      tail.setLead(i, value);
    }
  }

  /**
   * Set lengths.
   *
   * @tparam T1 Integer type.
   *
   * @param in Array assumed to have at least count() elements.
   */
  template<class T1>
  void setLengths(T1* in) {
    tail.setLengths(in);
    head.length = *(in + Tail::count());
  }

  /**
   * Set strides.
   *
   * @tparam T1 Integer type.
   *
   * @param in Array assumed to have at least count() elements.
   */
  template<class T1>
  void setStrides(T1* in) {
    tail.setStrides(in);
    head.stride = *(in + Tail::count());
  }

  /**
   * Get leads.
   *
   * @tparam T1 Integer type.
   *
   * @param in Array assumed to have at least count() elements.
   */
  template<class T1>
  void setLeads(T1* in) {
    tail.setLeads(in);
    head.lead = *(in + Tail::count());
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
  size_t size() const {
    return tail.size() * head.length;
  }

  /**
   * Product of all leads.
   */
  size_t volume() const {
    return tail.volume() * head.lead;
  }

  /**
   * Size of contiguous blocks.
   */
  size_t block() const {
    if (head.stride == 1) {
      if (head.lead == head.length) {
        return tail.block() * head.length;
      } else {
        return head.length;
      }
    } else {
      return 1;
    }
  }

  /**
   * Serial offset for a view.
   */
  template<class View>
  ptrdiff_t serial(const View& o) const {
    return tail.serial(o.tail) * head.lead + head.stride * o.head.offset;
  }

  /**
   * Are all elements stored contiguously in memory?
   */
  bool contiguous() const {
    return size() == 1 || (count() == 1 && head.stride == 1)
        || volume() == size();
  }
  //@}

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
  NonemptyFrame<Tail,Head>& operator*=(const ptrdiff_t n) {
    head *= n;
    return *this;
  }

  /**
   * Multiply stride of rightmost dimension.
   *
   * @todo Take index instead of n.
   */
  auto operator*(const ptrdiff_t n) const {
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
