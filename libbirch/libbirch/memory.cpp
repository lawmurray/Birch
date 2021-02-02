/**
 * @file
 */
#include "libbirch/memory.hpp"

#include "libbirch/Atomic.hpp"
#include "libbirch/Pool.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/Shared.hpp"
#include "libbirch/Marker.hpp"
#include "libbirch/Scanner.hpp"
#include "libbirch/Collector.hpp"
#include "libbirch/Spanner.hpp"
#include "libbirch/Bridger.hpp"
#include "libbirch/Copier.hpp"

/**
 * Type for object lists in cycle collection.
 */
using object_list = std::vector<libbirch::Any*,libbirch::Allocator<libbirch::Any*>>;

/**
 * Get the possible roots list for the current thread.
 */
static object_list& get_thread_possible_roots() {
  static std::vector<object_list,libbirch::Allocator<object_list>> objects(
      libbirch::get_max_threads());
  return objects[libbirch::get_thread_num()];
}

/**
 * Get the unreachable list for the current thread.
 */
static object_list& get_thread_unreachable() {
  static std::vector<object_list,libbirch::Allocator<object_list>> objects(
      libbirch::get_max_threads());
  return objects[libbirch::get_thread_num()];
}

/**
 * Make the heap.
 */
static char* make_heap() {
  #ifdef DISABLE_MEMORY_POOL
  return nullptr;
  #else
  /* determine a preferred size of the heap based on total physical memory */
  size_t size = sysconf(_SC_PAGE_SIZE);
  size_t npages = sysconf(_SC_PHYS_PAGES);
  size_t n = 8u*npages*size;

  /* attempt to allocate this amount, successively halving until
   * successful */
  void* ptr = nullptr;
  int res = 0;
  do {
    res = posix_memalign(&ptr, 64ull, n);
    n >>= 1;
  } while (res > 0 && n > 0u);
  assert(ptr);

  return (char*)ptr;
  #endif
}

/**
 * Get the heap.
 */
inline libbirch::Atomic<char*>& heap() {
  static libbirch::Atomic<char*> heap(make_heap());
  return heap;
}

/**
 * Get the `i`th pool.
 */
inline libbirch::Pool& pool(const int i) {
  static libbirch::Pool* pools =
      new libbirch::Pool[64*libbirch::get_max_threads()];
  return pools[i];
}

/**
 * For an allocation size, determine the index of the pool to which it
 * belongs.
 *
 * @param n Number of bytes.
 *
 * @return Pool index.
 *
 * Pool sizes are multiples of 8 bytes up to 64 bytes, and powers of two
 * thereafter.
 */
inline int bin(const size_t n) {
  assert(n > 0ull);
  int result = 0;
  #ifdef HAVE___BUILTIN_CLZLL
  if (n > 64ull) {
    /* __builtin_clzll undefined for zero argument */
    result = 64 - __builtin_clzll((n - 1ull) >> 6ull);
  }
  #else
  while (((n - 1ull) >> (6 + result)) > 0) {
    ++result;
  }
  #endif
  assert(0 <= result && result <= 63);
  return result;
}

/**
 * Determine the size for a given bin.
 */
inline size_t unbin(const int i) {
  return 64ull << i;
}

void* libbirch::allocate(const size_t n) {
  assert(n > 0u);

  #ifdef DISABLE_MEMORY_POOL
  return std::malloc(n);
  #else
  int tid = get_thread_num();
  int i = bin(n);       // determine which pool
  auto ptr = pool(64*tid + i).pop();  // attempt to reuse from this pool
  if (!ptr) {           // otherwise allocate new
    size_t m = unbin(i);
    ptr = (heap() += m) - m;
  }
  assert(ptr);
  return ptr;
  #endif
}

void libbirch::deallocate(void* ptr, const size_t n, const int tid) {
  assert(ptr);
  assert(n > 0u);
  assert(tid < get_max_threads());

  #ifdef DISABLE_MEMORY_POOL
  std::free(ptr);
  #else
  int i = bin(n);
  pool(64*tid + i).push(ptr);
  #endif
}

void* libbirch::reallocate(void* ptr1, const size_t n1, const int tid1,
    const size_t n2) {
  assert(ptr1);
  assert(n1 > 0u);
  assert(tid1 < get_max_threads());
  assert(n2 > 0u);

  #ifdef DISABLE_MEMORY_POOL
  return std::realloc(ptr1, n2);
  #else
  int i1 = bin(n1);
  int i2 = bin(n2);
  void* ptr2 = ptr1;
  if (i1 != i2) {
    /* can't continue using current allocation */
    ptr2 = allocate(n2);
    if (ptr1 && ptr2) {
      std::memcpy(ptr2, ptr1, std::min(n1, n2));
    }
    deallocate(ptr1, n1, tid1);
  }
  return ptr2;
  #endif
}

void libbirch::register_possible_root(Any* o) {
  assert(o);
  o->incShared();
  get_thread_possible_roots().emplace_back(o);
}

void libbirch::register_unreachable(Any* o) {
  assert(o);
  o->incShared();
  get_thread_unreachable().emplace_back(o);
}

void libbirch::collect() {
  #pragma omp parallel num_threads(get_max_threads())
  {
    /* mark */
    auto& possible_roots = get_thread_possible_roots();
    for (auto& o : possible_roots) {
      if (o) {
        if (o->isPossibleRoot()) {
          Marker visitor;
          visitor.visit(o);
        } else {
          o->decShared();
          o = nullptr;
        }
      }
    }
    #pragma omp barrier

    /* scan */
    for (auto& o : possible_roots) {
      if (o) {
        Scanner visitor;
        visitor.visit(o);
      }
    }
    #pragma omp barrier

    /* collect */
    for (auto& o : possible_roots) {
      if (o) {
        Collector visitor;
        visitor.visit(o);
        o->decShared();
        o = nullptr;
      }
    }
    possible_roots.clear();
    #pragma omp barrier

    /* destroy the objects indicated during collect */
    auto& unreachable = get_thread_unreachable();
    for (auto& o : unreachable) {
      o->destroy();
      o->decShared();  // removes last memo count, causing deallocate()
    }

    unreachable.clear();
  }
}

void libbirch::trim(Any* o) {
  auto& possible_roots = get_thread_possible_roots();
  while (!possible_roots.empty()) {
    auto ptr = possible_roots.back();
    if (ptr == o || !ptr->isPossibleRoot()) {
      possible_roots.pop_back();
      ptr->decShared();
    } else {
      return;
    }
  }
}
