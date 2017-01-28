/**
 * @file
 */
#include "bi/data/group.hpp"

bi::MemoryGroup bi::childGroup(const MemoryGroup& parent, const char* name) {
  return MemoryGroup();
}

bi::NetCDFGroup bi::childGroup(const NetCDFGroup& parent, const char* name) {
  if (parent.mode == NEW || parent.mode == REPLACE) {
    return parent.createGroup(name);
  } else {
    return parent.mapGroup(name);
  }
}
