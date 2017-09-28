/**
 * @file
 */
#include "bi/lib/Heap.hpp"

#include "bi/lib/Object.hpp"

bi::Heap::Heap() {
  //
}

bi::Heap::Heap(const Heap& o) :
    heap(o.heap) {
  ++fiberGen;
}

bi::Heap::~Heap() {
  //
}

bi::Heap& bi::Heap::operator=(const Heap& o) {
  heap = o.heap;
  ++fiberGen;
  return *this;
}

bi::Object* bi::Heap::get(const size_t index) {
  assert(index < heap.size());
  assert(heap[index]->getIndex() == index);
  return heap[index];
}

void bi::Heap::set(const size_t index, Object* raw) {
  assert(index < heap.size());
  raw->setGen(fiberGen);
  raw->setIndex(index);
  heap[index] = raw;
}

size_t bi::Heap::put(Object* raw) {
  heap.push_back(raw);
  size_t index = heap.size() - 1;
  raw->setGen(fiberGen);
  raw->setIndex(index);
  return index;
}
