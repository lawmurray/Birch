/**
 * @file
 */
#include "bi/lib/global.hpp"

#include "bi/lib/Heap.hpp"

static std::random_device rd; // for initial setup of random number generator

bi::Heap* bi::fiberHeap = new bi::Heap();
std::mt19937_64 bi::rng(rd());
