/**
 * @file
 */
#include "libbirch/libbirch.hpp"

extern "C" int bi_present() {
  return 0;
}

bi::Range<> bi::make_range(const ptrdiff_t start, const ptrdiff_t end) {
  ptrdiff_t length = std::max(ptrdiff_t(0), end - start + 1);
  return Range<>(start, length);
}

bi::EmptyFrame bi::make_frame() {
  return EmptyFrame();
}

bi::NonemptyFrame<bi::Span<>,bi::EmptyFrame> bi::make_frame(
    const size_t arg) {
  auto tail = EmptyFrame();
  auto head = Span<>(arg, tail.volume());
  return NonemptyFrame<Span<>,EmptyFrame>(head, tail);
}

bi::EmptyView bi::make_view() {
  return EmptyView();
}

bi::NonemptyView<bi::Index<>,bi::EmptyView> bi::make_view(
    const ptrdiff_t arg) {
  auto head = Index<>(arg);
  auto tail = EmptyView();
  return NonemptyView<Index<>,EmptyView>(head, tail);
}
