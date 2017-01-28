/**
 * @file
 */
#pragma once
#include "bi/data/MemoryGroup.hpp"
#include "bi/data/NetCDFGroup.hpp"

namespace bi {
/*
 * Create a child group.
 */
MemoryGroup childGroup(const MemoryGroup& parent, const char* name = nullptr);
NetCDFGroup childGroup(const NetCDFGroup& parent, const char* name = nullptr);

}
