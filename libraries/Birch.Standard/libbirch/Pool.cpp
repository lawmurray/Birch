/**
 * @file
 */
#include "libbirch/Pool.hpp"

#include "libbirch/memory.hpp"

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
  lock.set();
  setNext(block, top);
  top = block;
  lock.unset();
}

void* libbirch::Pool::getNext(void* block) {
  return (block) ? *reinterpret_cast<void**>(block) : nullptr;
}

void libbirch::Pool::setNext(void* block, void* value) {
  assert(block);
  *reinterpret_cast<void**>(block) = value;
}
