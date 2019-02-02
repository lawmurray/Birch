/**
 * @file
 */
#include "libbirch/Pool.hpp"

bi::Pool::Pool() :
    top(nullptr) {
  //
}

bool bi::Pool::empty() const {
  return !top;
}

void* bi::Pool::pop() {
  //lock.keep();
  auto result = top;
  top = getNext(result);
  //lock.unkeep();
  return result;
}

void bi::Pool::push(void* block) {
  assert(bufferStart <= block && block < bufferStart + bufferSize);
  //lock.keep();
  setNext(block, top);
  top = block;
  //lock.unkeep();
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
