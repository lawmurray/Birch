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
 * Possible roots list for each thread.
 */
static thread_local std::vector<libbirch::Any*> possible_roots;

/**
 * Unreachable list for each thread.
 */
static thread_local std::vector<libbirch::Any*> unreachable;

/**
 * Biconnected flag for each thread.
 */
static thread_local bool biconnected_flag = false;

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
    ptr = std::malloc(m);
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
  possible_roots.push_back(o);
}

void libbirch::register_unreachable(Any* o) {
  unreachable.push_back(o);
}

void libbirch::collect() {
  /* concatenates the possible roots list of each thread into a single list,
   * then distributes the passes over this list between all threads; this
   * improves load balancing over each thread operating only on its original
   * list */

  auto nthreads = get_max_threads();
  std::vector<libbirch::Any*> all_possible_roots;   // concatenated list of possible roots
  std::vector<int> starts(nthreads); // start indices in concatenated list
  std::vector<int> sizes(nthreads);  // sizes in concatenated list

  #pragma omp parallel
  {
    auto tid = get_thread_num();

    /* objects can be added to the possible roots list during normal
     * execution, but not removed, although they may be flagged as no longer
     * being a possible root; remove such objects first */
    int size = 0;
    for (int i = 0; i < (int)possible_roots.size(); ++i) {
      auto o = possible_roots[i];
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
      #ifdef __cpp_lib_parallel_algorithm
      std::exclusive_scan(sizes.begin(), sizes.end(), starts.begin(), 0);
      #else
      starts[0] = 0;
      for (int i = 1; i < nthreads; ++i) {
        starts[i] = starts[i - 1] + sizes[i - 1];
      }
      #endif
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
    for (int i = 0; i < (int)all_possible_roots.size(); ++i) {
      auto o = all_possible_roots[i];
      Marker visitor;
      visitor.visit(o);
    }
    #pragma omp barrier

    /* scan/reach pass */
    #pragma omp for schedule(guided)
    for (int i = 0; i < (int)all_possible_roots.size(); ++i) {
      auto o = all_possible_roots[i];
      Scanner visitor;
      visitor.visit(o);
    }
    #pragma omp barrier

    /* collect pass */
    #pragma omp for schedule(guided)
    for (int i = 0; i < (int)all_possible_roots.size(); ++i) {
      auto o = all_possible_roots[i];
      Collector visitor;
      visitor.visit(o);
    }
    #pragma omp barrier

    /* finally, destroy objects determined unreachable */
    for (int i = 0; i < (int)unreachable.size(); ++i) {
      auto o = unreachable[i];
      o->destroy_();
      o->deallocate_();
    }
    unreachable.clear();
  }
}

bool libbirch::biconnected_copy(const bool toggle) {
  if (toggle) {
    biconnected_flag = !biconnected_flag;
  }
  return biconnected_flag;
}
