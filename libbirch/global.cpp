/**
 * @file
 */
#include "libbirch/global.hpp"

#include "libbirch/Label.hpp"
#include "libbirch/SharedPtr.hpp"

static libbirch::Label* makeRootLabel() {
  static libbirch::SharedPtr<libbirch::Label> rootLabel(new libbirch::Label());
  return rootLabel.get();
}

libbirch::Label* libbirch::rootLabel(makeRootLabel());
