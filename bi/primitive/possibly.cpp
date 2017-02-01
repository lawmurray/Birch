/**
 * @file
 */
#include "bi/primitive/possibly.hpp"

#include <algorithm>

bi::possibly bi::possibly::operator&&(const possibly& o) const {
  return possibly(std::min(state, o.state));
}

bi::possibly bi::possibly::operator||(const possibly& o) const {
  return possibly(std::max(state, o.state));
}

bi::possibly bi::possibly::operator&&(const bool& o) const {
  return possibly(std::min(state, possibly(o).state));
}

bi::possibly bi::possibly::operator||(const bool& o) const {
  return possibly(std::max(state, possibly(o).state));
}

bi::possibly bi::possibly::operator!() const {
  return possibly(DEFINITE - state);
}
