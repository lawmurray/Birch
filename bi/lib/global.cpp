/**
 * @file
 */
#include "bi/lib/global.hpp"

#include "bi/lib/Heap.hpp"

static std::random_device rd;

bi::Heap* bi::fiberHeap = new bi::Heap();
size_t bi::fiberGen = 0;
std::mt19937_64 bi::rng(rd());
