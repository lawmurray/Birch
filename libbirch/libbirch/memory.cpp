/**
 * @file
 */
#include "libbirch/memory.hpp"

#include "libbirch/thread.hpp"
#include "libbirch/Atomic.hpp"
#include "libbirch/Pool.hpp"
#include "libbirch/Allocator.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/Shared.hpp"
#include "libbirch/Marker.hpp"
#include "libbirch/Scanner.hpp"
#include "libbirch/Collector.hpp"
#include "libbirch/Spanner.hpp"

/**
 * Type for object lists in cycle collection.
 */
using object_list = std::vector<libbirch::Any*>;

/**
 * Type for size lists in cycle collection.
 */
using size_list = std::vector<int,libbirch::Allocator<int>>;

/**
 * Get the `i`th possible roots list for the current thread.
 */
static object_list& get_possible_roots(const int i) {
  static std::vector<object_list> objects(libbirch::get_max_threads());
  return objects[i];
}

/**
 * Get the `i`th unreachable list.
 */
static object_list& get_unreachable(const int i) {
  static std::vector<object_list> objects(libbirch::get_max_threads());
  return objects[i];
}

#ifdef ENABLE_MEMORY_POOL
/**
 * Make the heap.
 */
static char* make_heap() {
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
}
#endif

#ifdef ENABLE_MEMORY_POOL
/**
 * Get the heap.
 */
static libbirch::Atomic<char*>& get_heap() {
  static libbirch::Atomic<char*> heap(make_heap());
  return heap;
}
#endif

#ifdef ENABLE_MEMORY_POOL
/**
 * Get the `i`th pool.
 */
static libbirch::Pool& get_pool(const int i) {
  static libbirch::Pool* pools =
      new libbirch::Pool[64*libbirch::get_max_threads()];
  return pools[i];
}
#endif

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

  #ifdef ENABLE_MEMORY_POOL
  int tid = get_thread_num();
  int i = bin(n);       // determine which pool
  auto ptr = get_pool(64*tid + i).take();  // attempt to reuse from this pool
  if (!ptr) {           // otherwise allocate new
    size_t m = unbin(i);
    ptr = (get_heap() += m) - m;
  }
  assert(ptr);
  return ptr;
  #else
  return std::malloc(n);
  #endif
}

void libbirch::deallocate(void* ptr, const size_t n, const int tid) {
  assert(ptr);
  assert(n > 0u);
  assert(tid < get_max_threads());

  #ifdef ENABLE_MEMORY_POOL
  int i = bin(n);
  auto& pool = get_pool(64*tid + i);
  if (tid == get_thread_num()) {
    /* this thread owns the associated pool, return the allocation to the free
     * list */
    pool.free(ptr);
  } else {
    /* this thread does not own the associated pool, return the allocation to
     * the pend list */
    pool.pend(ptr);
  }
  #else
  std::free(ptr);
  #endif
}

void* libbirch::reallocate(void* ptr1, const size_t n1, const int tid1,
    const size_t n2) {
  assert(ptr1);
  assert(n1 > 0u);
  assert(tid1 < get_max_threads());
  assert(n2 > 0u);

  #ifdef ENABLE_MEMORY_POOL
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
  #else
  return std::realloc(ptr1, n2);
  #endif
}

void libbirch::register_possible_root(Any* o) {
  get_possible_roots(get_thread_num()).push_back(o);
}

void libbirch::register_unreachable(Any* o) {
  get_unreachable(get_thread_num()).push_back(o);
}

void libbirch::collect() {
  /* concatenates the possible roots list of each thread into a single list,
   * then distributes the passes over this list between all threads; this
   * improves load balancing over each thread operating only on its original
   * list */

  auto nthreads = get_max_threads();
  object_list all_possible_roots;  // concatenated list of possible roots
  size_list starts(nthreads), sizes(nthreads);  // ranges in concatenated list

  #pragma omp parallel
  {
    auto tid = get_thread_num();
    auto& possible_roots = get_possible_roots(tid);

    /* objects can be added to the possible roots list during normal
     * execution, but not removed, although they may be flagged as no longer
     * being a possible root; remove such objects first */
    int size = 0;
    for (auto o: possible_roots) {
      if (o->isPossibleRoot_()) {
        possible_roots[size++] = o;
      } else if (o->numShared_() == 0) {
        o->deallocate_();  // deallocation was deferred until now
      } else {
        o->unbuffer_();  // not a root, mark as no longer in the buffer
      }
    }
    possible_roots.resize(size);
    sizes[tid] = size;
    #pragma omp barrier

    /* a single thread now sets up the concatenated list of possible roots */
    #pragma omp single
    {
      std::exclusive_scan(sizes.begin(), sizes.end(), starts.begin(), 0);
      all_possible_roots.resize(starts.back() + sizes.back());
    }
    #pragma omp barrier

    /* all threads copy into the concatenated list of possible roots */
    std::copy(possible_roots.begin(), possible_roots.end(),
        all_possible_roots.begin() + starts[tid]);
    possible_roots.clear();
    #pragma omp barrier

    /* mark pass */
    #pragma omp for schedule(guided)
    for (auto o : all_possible_roots) {
      Marker visitor;
      visitor.visit(o);
    }
    #pragma omp barrier

    /* scan/reach pass */
    #pragma omp for schedule(guided)
    for (auto o : all_possible_roots) {
      Scanner visitor;
      visitor.visit(o);
    }
    #pragma omp barrier

    /* collect pass */
    #pragma omp for schedule(guided)
    for (auto o : all_possible_roots) {
      Collector visitor;
      visitor.visit(o);
    }
    #pragma omp barrier

    /* finally, destroy objects determined unreachable */
    auto& unreachable = get_unreachable(tid);
    for (auto o : unreachable) {
      o->destroy_();
      o->deallocate_();
    }
    unreachable.clear();
  }
}

bool libbirch::biconnected_copy(const bool toggle) {
  /* don't use std::vector<bool> here, as it is often specialized as a bitset,
   * which is not thread safe */
  static std::vector<int8_t> flags(libbirch::get_max_threads(), 0);
  auto tid = libbirch::get_thread_num();
  if (toggle) {
    flags[tid] = !flags[tid];
  }
  return flags[tid];
}
