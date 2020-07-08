/**
 * @file
 */
#include "libbirch/memory.hpp"

#include "libbirch/Any.hpp"
#include "libbirch/Label.hpp"
#include "libbirch/Shared.hpp"

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
 * Create the heap.
 */
static char* make_heap() {
  #if !ENABLE_MEMORY_POOL
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
 * Create the root label.
 */
static auto make_root_label() {
  auto label = new libbirch::Label();
  label->incShared();
  return label;
}

libbirch::ExitBarrierLock libbirch::finish_lock;
libbirch::ExitBarrierLock libbirch::freeze_lock;

libbirch::Atomic<char*> libbirch::heap(make_heap());
libbirch::Label* const libbirch::root_label(make_root_label());

libbirch::Pool& libbirch::pool(const int i) {
  static libbirch::Pool* pools = new libbirch::Pool[64*get_max_threads()];
  return pools[i];
}

void* libbirch::allocate(const size_t n) {
  assert(n > 0u);

  #if !ENABLE_MEMORY_POOL
  return std::malloc(n);
  #else
  int tid = get_thread_num();
  int i = bin(n);       // determine which pool
  auto ptr = pool(64*tid + i).pop();  // attempt to reuse from this pool
  if (!ptr) {           // otherwise allocate new
    size_t m = unbin(i);
    ptr = (heap += m) - m;
  }
  assert(ptr);
  return ptr;
  #endif
}

void libbirch::deallocate(void* ptr, const size_t n, const int tid) {
  assert(ptr);
  assert(n > 0u);
  assert(tid < get_max_threads());

  #if !ENABLE_MEMORY_POOL
  std::free(ptr);
  #else
  int i = bin(n);
  pool(64*tid + i).push(ptr);
  #endif
}

void libbirch::deallocate(void* ptr, const unsigned n, const int tid) {
  assert(ptr);
  assert(n > 0u);
  assert(tid < get_max_threads());

  #if !ENABLE_MEMORY_POOL
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

  #if !ENABLE_MEMORY_POOL
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
  o->incMemo();
  get_thread_possible_roots().emplace_back(o);
}

void libbirch::register_unreachable(Any* o) {
  assert(o);
  o->incMemo();
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
          o->mark();
        } else {
          o->decMemo();
          o = nullptr;
        }
      }
    }
    #pragma omp barrier

    /* scan */
    for (auto& o : possible_roots) {
      if (o) {
        o->scan();
      }
    }
    #pragma omp barrier

    /* collect */
    for (auto& o : possible_roots) {
      if (o) {
        o->collect();
        o->decMemo();
        o = nullptr;
      }
    }
    possible_roots.clear();

    /* destroy the objects indicated during collect */
    auto& unreachable = get_thread_unreachable();
    for (auto& o : unreachable) {
      o->destroy();
      o->decMemo();
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
      ptr->decMemo();
    } else {
      return;
    }
  }
}
