/**
 * @file
 */
#if ENABLE_MEMORY_POOL
#include "libbirch/Pool.hpp"

libbirch::Pool::Pool() :
    top(nullptr) {
  //
}

bool libbirch::Pool::empty() const {
  return !top;
}

void* libbirch::Pool::pop() {
  lock.keep();
  auto result = top;
  top = getNext(result);
  lock.unkeep();
  return result;
}

void libbirch::Pool::push(void* block) {
  assert(bufferStart <= block && block < bufferStart + bufferSize);
  lock.keep();
  setNext(block, top);
  top = block;
  lock.unkeep();
}

void* libbirch::Pool::getNext(void* block) {
  assert(
      !block || (bufferStart <= block && block < bufferStart + bufferSize));

  return (block) ? *reinterpret_cast<void**>(block) : nullptr;
}

void libbirch::Pool::setNext(void* block, void* value) {
  assert(
      !block || (bufferStart <= block && block < bufferStart + bufferSize));
  assert(
      !value || (bufferStart <= value && value < bufferStart + bufferSize));

  if (block) {
    *reinterpret_cast<void**>(block) = value;
  }
}

#endif
