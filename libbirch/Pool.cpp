/**
 * @file
 */
#if ENABLE_MEMORY_POOL
#include "libbirch/Pool.hpp"

#include "libbirch/memory.hpp"

libbirch::Pool& libbirch::pool(const int i) {
  static libbirch::Pool* pools = new libbirch::Pool[64*get_max_threads()];
  return pools[i];
}

libbirch::Pool::Pool() :
    top(nullptr) {
  //
}

bool libbirch::Pool::empty() const {
  return !top;
}

void* libbirch::Pool::pop() {
  lock.set();
  auto result = top;
  top = getNext(result);
  lock.unset();
  return result;
}

void libbirch::Pool::push(void* block) {
  assert(buffer_start <= block && block < buffer_start + buffer_size);
  lock.set();
  setNext(block, top);
  top = block;
  lock.unset();
}

void* libbirch::Pool::getNext(void* block) {
  assert(
      !block || (buffer_start <= block && block < buffer_start + buffer_size));

  return (block) ? *reinterpret_cast<void**>(block) : nullptr;
}

void libbirch::Pool::setNext(void* block, void* value) {
  assert(block);
  assert(buffer_start <= block && block < buffer_start + buffer_size);
  assert(
      !value || (buffer_start <= value && value < buffer_start + buffer_size));

  *reinterpret_cast<void**>(block) = value;
}

#endif
