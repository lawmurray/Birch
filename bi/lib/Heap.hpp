/**
 * @file
 */
#pragma once

#include <gc.h>
#include <gc/gc_allocator.h>

#include <vector>

namespace bi {
class Markable;
template<class T> class Pointer;

/**
 * Heap for garbage-collected allocations.
 *
 * The heap performs two types of allocations:
 *
 *   @li global allocations,
 *   @li coroutine-local allocations.
 *
 * The former are garbage-collected conservatively using the Boehm garbage
 * collector. The latter are garbage-collected precisely using a custom
 * mark-sweep garbage collector, made possible by the transparent stack of
 * coroutines in Birch. Furthermore, coroutine-local allocations are
 * copyable and relocatable and, in combination with Pointer, use
 * copy-on-write semantics. This facilitates lightweight copying of coroutines
 * in a manner analagous to @c fork() for processes, and their portable
 * relocation in a distributed setting.
 */
class Heap {
public:
  /**
   * Constructor.
   *
   * @param Is this the global heap?
   */
  Heap(const bool global = true);

  /**
   * Destructor.
   */
  ~Heap();

  /**
   * Construct object.
   *
   * @param args Constructor arguments.
   */
  template<class Type, class ... Args>
  Pointer<Type> make(Args ... args);

  /**
   * Allocate memory for array of value type.
   *
   * @tparam Type Element type.
   *
   * @param[out] ptr Pointer to start of allocated buffer.
   * @param size Number of bytes to allocate.
   */
  template<class Type>
  void allocate(Pointer<Type>& ptr, const size_t size);

  /**
   * Allocate memory for array of pointer type.
   *
   * @tparam Type Element type.
   *
   * @param[out] ptr Pointer to start of allocated buffer.
   * @param size Number of bytes to allocate.
   */
  template<class Type>
  void allocate(Pointer<Pointer<Type>>& ptr, const size_t size);

  /**
   * Initialise memory.
   *
   * @param ptr Memory address to initialise.
   * @param args Constructor arguments.
   */
  template<class Type, class ... Args>
  void initialise(Type& o, Args ... args);

  /**
   * Initialise memory.
   *
   * @param ptr Memory address to initialise.
   * @param args Constructor arguments.
   */
  template<class Type, class ... Args>
  void initialise(Pointer<Type>& o, Args ... args);

  /**
   * Convert a smart pointer to a raw pointer.
   */
  template<class Type>
  Type* get(const Pointer<Type>& ptr) const;

  /**
   * Unmark all allocations.
   */
  void unmark();

  /**
   * Free all unreachable allocations.
   */
  void sweep();

private:
  /**
   * Items on the heap.
   */
  std::vector<Markable*,gc_allocator<Markable*>> items;

  /**
   * Is this a global heap?
   */
  bool global;
};

/**
 * The global heap.
 */
extern Heap heap;
}

#include "bi/lib/Pointer.hpp"

template<class Type, class ... Args>
bi::Pointer<Type> bi::Heap::make(Args ... args) {
  Type* ptr = new (GC_MALLOC(sizeof(Type))) Type(args...);
  if (global) {
    return Pointer<Type>(ptr);
  } else {
    items.push_back(ptr);
    return Pointer<Type>(items.size() - 1);
  }
}

template<class Type>
void bi::Heap::allocate(Pointer<Type>& ptr, const size_t size) {
  auto address = static_cast<Type*>(GC_MALLOC_ATOMIC(size));
  // ^ buffer cannot itself contain pointers, so GC_MALLOC_ATOMIC can be used
  //   to prevent the garbage collector sweeping through it

  if (global) {
    return Pointer<Type>(address);
  } else {
    items.push_back(address);
    return Pointer<Type>(items.size() - 1);
  }
}

template<class Type>
void bi::Heap::allocate(Pointer<Pointer<Type>>& ptr, const size_t size) {
  auto address = static_cast<Pointer<Type>*>(GC_MALLOC(size));
  if (global) {
    ptr = address;
  } else {
    items.push_back(address);
    ptr = items.size() - 1;
  }
}

template<class Type, class ... Args>
void bi::Heap::initialise(Type& o, Args ... args) {
  new (&o) Type(args...);
}

template<class Type, class ... Args>
void bi::Heap::initialise(Pointer<Type>& o, Args ... args) {
  Type* ptr = new (GC_MALLOC(sizeof(Type))) Type(args...);
  if (global) {
    new (&o) Pointer<Type>(ptr);
  } else {
    items.push_back(ptr);
    new (&o) Pointer<Type>(items.size() - 1);
  }
}

template<class Type>
Type* bi::Heap::get(const Pointer<Type>& ptr) const {
  assert(!global || ptr.index < 0);
  return ptr.index >= 0 ? dynamic_cast<Type*>(items[ptr.index]) : ptr.ptr;
}
