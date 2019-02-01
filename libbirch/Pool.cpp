/**
 * @file
 */
#include "libbirch/Pool.hpp"

void* bi::Pool::pop() {
  stack_t expected = stack.load();
  stack_t desired = { getNext(expected.top), expected.count + 1u };
  while (expected.top
      && !stack.compare_exchange_weak(expected, desired,
          std::memory_order_relaxed)) {
    desired = {getNext(expected.top), expected.count + 1u};
  }
  return expected.top;
}

void bi::Pool::push(void* block) {
  assert(bufferStart <= block && block < bufferStart + bufferSize);

  stack_t expected = stack.load();
  stack_t desired = { block, expected.count + 1u };
  setNext(block, expected.top);
  while (!stack.compare_exchange_weak(expected, desired,
      std::memory_order_relaxed)) {
    desired.count = expected.count + 1u;
    setNext(block, expected.top);
  }
}

void* bi::Pool::getNext(void* block) {
  assert(
      !block || (bufferStart <= block && block < bufferStart + bufferSize));

  return (block) ? *reinterpret_cast<void**>(block) : nullptr;
}

void bi::Pool::setNext(void* block, void* value) {
  assert(
      !block || (bufferStart <= block && block < bufferStart + bufferSize));
  assert(
      !value || (bufferStart <= value && value < bufferStart + bufferSize));

  if (block) {
    *reinterpret_cast<void**>(block) = value;
  }
}
