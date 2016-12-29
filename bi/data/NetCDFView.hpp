/**
 * @file
 */
#pragma once

#include "bi/data/Frame.hpp"
#include "bi/data/View.hpp"

#include <vector>
#include <algorithm>

namespace bi {
/**
 * NetCDF view.
 *
 * @ingroup library
 *
 * Describes the active elements over an array for NetCDFGroup.
 */
struct NetCDFView {
  /**
   * Constructor.
   *
   * @param frame Frame.
   */
  template<class Frame = EmptyFrame>
  NetCDFView(const Frame& frame = EmptyFrame());

  /**
   * Convolve another view into this one.
   */
  template<class View>
  void convolve(const View& view);

  /**
   * Offsets along dimensions.
   */
  std::vector<size_t> offsets;

  /**
   * Lengths along dimensions.
   */
  std::vector<size_t> lengths;

  /**
   * Strides along dimensions.
   */
  std::vector<size_t> strides;

  /**
   * Which dimensions are specified by indexes rather than ranges?
   */
  std::vector<bool> indices;

protected:
  template<class Tail, int_t offset_value, int_t length_value,
      int_t stride_value>
  void convolve(
      const NonemptyView<Tail,Range<offset_value,length_value,stride_value>>& view,
      const int d);
  template<class Tail, int_t offset_value>
  void convolve(const NonemptyView<Tail,Index<offset_value>>& view,
      const int d);
  void convolve(const EmptyView& view, const int d);
};
}

template<class Frame>
bi::NetCDFView::NetCDFView(const Frame& frame) :
    offsets(frame.count()),
    lengths(frame.count()),
    strides(frame.count()),
    indices(frame.count()) {
  std::fill(offsets.begin(), offsets.end(), 0);
  frame.lengths(lengths.data());
  frame.strides(strides.data());
  std::fill(indices.begin(), indices.end(), false);
}

template<class View>
void bi::NetCDFView::convolve(const View& view) {
  convolve(view, lengths.size());
}

template<class Tail, bi::int_t offset_value, bi::int_t length_value,
    bi::int_t stride_value>
void bi::NetCDFView::convolve(
    const NonemptyView<Tail,Range<offset_value,length_value,stride_value>>& view,
    const int d) {
  /* pre-condition */
  assert(d > 0);

  if (indices[d - 1]) {
    convolve(view, d - 1);
  } else {
    offsets[d - 1] += view.head.offset * strides[d - 1];
    lengths[d - 1] = view.head.length;
    strides[d - 1] *= view.head.stride;

    convolve(view.tail, d - 1);
  }
}

template<class Tail, bi::int_t offset_value>
void bi::NetCDFView::convolve(
    const NonemptyView<Tail,Index<offset_value>>& view, const int d) {
  /* pre-condition */
  assert(d > 0);

  offsets[d - 1] = view.head.offset;
  lengths[d - 1] = 1;
  strides[d - 1] = 1;
  indices[d - 1] = true;

  convolve(view.tail, d - 1);
}
