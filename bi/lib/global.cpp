/**
 * @file
 */
#include "bi/lib/global.hpp"

#include "bi/lib/AllocationMap.hpp"

static std::random_device rd;

bi::AllocationMap* bi::fiberAllocationMap = new bi::AllocationMap();
std::mt19937_64 bi::rng(rd());
