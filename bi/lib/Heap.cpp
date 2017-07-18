/**
 * @file
 */
#include "bi/lib/Heap.hpp"

#include "bi/lib/Markable.hpp"

bi::Heap bi::heap;

bi::Heap::Heap(const bool global) :
    global(global) {
  //
}

bi::Heap::~Heap() {
  unmark();
  sweep();
}

void bi::Heap::unmark() {
  for (auto o : items) {
    o->unmark();
  }
}

void bi::Heap::sweep() {
  for (auto o : items) {
    if (!o->isMarked()) {
      GC_FREE(o);
      o = nullptr;
    }
  }
}
