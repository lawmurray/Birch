/**
 * @file
 */
#include "bi/lib/Heap.hpp"

#include "bi/lib/Object.hpp"

bi::Heap::Heap() : gen(0) {
  //
}

bi::Heap::Heap(const Heap& o) :
    heap(o.heap),
    gen(o.gen + 1) {
  //
}

bi::Heap::~Heap() {
  //
}

bi::Heap& bi::Heap::operator=(const Heap& o) {
  heap = o.heap;
  gen = o.gen + 1;
  return *this;
}

bi::Object* bi::Heap::get(const size_t index) {
  assert(index < heap.size());
  assert(heap[index]->getIndex() == index);
  return heap[index];
}

void bi::Heap::set(const size_t index, Object* raw) {
  assert(index < heap.size());
  raw->setGen(gen);
  raw->setIndex(index);
  heap[index] = raw;
}

size_t bi::Heap::put(Object* raw) {
  heap.push_back(raw);
  size_t index = heap.size() - 1;
  raw->setGen(gen);
  raw->setIndex(index);
  return index;
}
