/**
 * @file
 */
#pragma once

#include "bi/data/StackGroup.hpp"
#include "bi/data/RefGroup.hpp"
#include "bi/data/HeapGroup.hpp"
#include "bi/data/NetCDFGroup.hpp"

namespace bi {
/*
 * Create a child group for a scalar.
 */
StackGroup childGroup(const StackGroup& parent, const char* name = nullptr);
RefGroup childGroup(const RefGroup& parent, const char* name = nullptr);
HeapGroup childGroup(const HeapGroup& parent, const char* name = nullptr);
NetCDFGroup childGroup(const NetCDFGroup& parent, const char* name = nullptr);

/*
 * Create a child group for an array.
 */
HeapGroup arrayGroup(const StackGroup& parent, const char* name = nullptr);
RefGroup arrayGroup(const RefGroup& parent, const char* name = nullptr);
HeapGroup arrayGroup(const HeapGroup& parent, const char* name = nullptr);
NetCDFGroup arrayGroup(const NetCDFGroup& parent, const char* name = nullptr);
}
