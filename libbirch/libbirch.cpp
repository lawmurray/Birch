/**
 * @file
 */
#include "libbirch/libbirch.hpp"

bi::Range<> bi::make_range(const int64_t start, const int64_t end) {
  int64_t length = std::max(int64_t(0), end - start + 1);
  return Range<>(start, length);
}

bi::EmptyFrame bi::make_frame() {
  return EmptyFrame();
}

bi::NonemptyFrame<bi::Span<>,bi::EmptyFrame> bi::make_frame(
    const int64_t arg) {
  auto tail = EmptyFrame();
  auto head = Span<>(arg, tail.volume());
  return NonemptyFrame<Span<>,EmptyFrame>(head, tail);
}

bi::EmptyView bi::make_view() {
  return EmptyView();
}

bi::NonemptyView<bi::Index<>,bi::EmptyView> bi::make_view(
    const int64_t arg) {
  auto head = Index<>(arg);
  auto tail = EmptyView();
  return NonemptyView<Index<>,EmptyView>(head, tail);
}
