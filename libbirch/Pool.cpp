/**
 * @file
 */
#include "libbirch/Pool.hpp"

bi::Pool::Pool() : stack({nullptr, nullptr}) {
  //
}

bool bi::Pool::empty() const {
  return !stack.load().top;
}

void* bi::Pool::pop() {
  stack_t expected = stack.load();
  stack_t desired = {expected.next, getNext(expected.next)};
  while (expected.top && !stack.compare_exchange_weak(expected, desired)) {
    desired = {expected.next, getNext(expected.next)};
  }
  return expected.top;
}

void bi::Pool::push(void* block) {
  stack_t expected = stack.load();
  stack_t desired = {block, expected.top};
  setNext(block, expected.top);
  while (!stack.compare_exchange_weak(expected, desired)) {
    desired.next = expected.top;
    setNext(block, expected.top);
  }
}

void* bi::Pool::getNext(void* block) {
  return (block) ? *reinterpret_cast<void**>(block) : nullptr;
}

void bi::Pool::setNext(void* block, void* value) {
  if (block) {
    *reinterpret_cast<void**>(block) = value;
  }
}
