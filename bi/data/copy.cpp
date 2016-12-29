/**
 * @file
 */
#include "bi/data/copy.hpp"

bi::int_t bi::common_length(const int_t a, const int_t b) {
  int_t a1 = a, b1 = b;
  while (a1 != b1) {
    a1 = a1 % b1;
    std::swap(a1, b1);
  }
  return a1;
}
