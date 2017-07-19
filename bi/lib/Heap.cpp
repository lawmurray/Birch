/**
 * @file
 */
#include "bi/lib/Heap.hpp"

#include "bi/lib/Object.hpp"

bi::Heap bi::heap;

bi::Heap::Heap(const bool global) :
    global(global) {
  //
}

bi::Heap::~Heap() {
  //
}

void bi::Heap::sweep() {
  auto item = items.begin();
  auto mark = marks.begin();
  while (item != items.end() && mark != marks.end()) {
    if (!*mark) {
      (*item)->disuse();
      //GC_FREE(*item);  // optional
      *item = nullptr;
    } else {
      *mark = false;  // clear for next time
    }
    ++item;
    ++mark;
  }
}
