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
  /* update fiber usage counts */
  for (auto o : heap) {
    o->use();
  }
  /// @todo For multiple copies of the same fiber (common use case), could
  /// update usage counts just once
}

bi::Heap::~Heap() {
  /* update fiber usage counts */
  for (auto o : heap) {
    o->disuse();
  }
}

bi::Heap& bi::Heap::operator=(const Heap& o) {
  auto iter1 = heap.begin();
  auto iter2 = o.heap.begin();
  auto end1 = heap.end();
  auto end2 = o.heap.end();

  /* skip through common history */
  while (iter1 != end1 && iter2 != end2 && *iter1 == *iter2) {
    ++iter1;
    ++iter2;
  }

  /* disuse remaining allocations on the left of the assignment */
  while (iter1 != end1) {
    (*iter1)->disuse();
    ++iter1;
  }

  /* copy remaining allocations from the right side of the assignment */
  heap.resize(o.heap.size());
  iter1 = heap.begin() + std::distance(o.heap.begin(), iter2);
  while (iter2 != end2) {
    *iter1 = *iter2;
    (*iter1)->use();
    ++iter1;
    ++iter2;
  }
  assert(iter1 == heap.end());
  assert(iter2 == o.heap.end());

  return *this;
}

bi::Object* bi::Heap::get(const size_t index) {
  assert(index < heap.size());
  assert(heap[index]->getIndex() == index);
  return heap[index];
}

void bi::Heap::set(const size_t index, Object* raw) {
  assert(index < heap.size());
  raw->setIndex(index);
  heap[index] = raw;
}

size_t bi::Heap::put(Object* raw) {
  heap.push_back(raw);
  size_t index = heap.size() - 1;
  raw->setIndex(index);
  return index;
}
