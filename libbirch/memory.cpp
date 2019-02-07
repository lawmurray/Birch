/**
 * @file
 */
#include "libbirch/memory.hpp"

#include "libbirch/Memo.hpp"

#include <unistd.h>

std::atomic<char*> bi::buffer(heap());
char* bi::bufferStart;
size_t bi::bufferSize;
bi::Pool bi::pool[64*48];

/* from clone.hpp, put here rather than clone.cpp to ensure correct
 * initialization order of global variables */
static bi::SharedPtr<bi::Memo> rootMemo = bi::Memo::create();
bi::Memo* bi::currentContext = rootMemo.get();
bool bi::cloneUnderway = false;

char* bi::heap() {
#if !USE_MEMORY_POOL
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

  bufferStart = (char*)ptr;
  bufferSize = n;

  return (char*)ptr;
#endif
}

void* bi::allocate(const size_t n) {
  assert(n > 0u);
#if !USE_MEMORY_POOL
  return std::malloc(n);
#else
  int i = bin(n);       // determine which pool
  ptr = pool[64*tid + i].pop();  // attempt to reuse from this pool
  if (!ptr) {           // otherwise allocate new
    size_t m = unbin(i);
    size_t r = (m < 64u) ? 64u : m;
    // ^ minimum allocation 64 bytes to maintain alignment
    ptr = buffer.fetch_add(r, std::memory_order_seq_cst);
    assert((char*)ptr + r <= bufferStart + bufferSize); // otherwise out of memory
    if (m < 64u) {
      /* add extra bytes as a separate allocation to the pool for
       * reuse another time */
      pool[64*tid + bin(64u - m)].push((char*)ptr + m);
    }
  }
  assert(ptr);
  return ptr;
#endif
}

void bi::deallocate(void* ptr, const size_t n) {
  assert(ptr && n > 0u);
#if !USE_MEMORY_POOL
  std::free(ptr);
#else
  int i = bin(n);
  pool[64*tid + i].push(ptr);
#endif
}

void bi::deallocate(void* ptr, const unsigned n) {
  assert(ptr && n > 0u);
#if !USE_MEMORY_POOL
  std::free(ptr);
#else
  int i = bin(n);
  pool[64*tid + i].push(ptr);
#endif
}

void* bi::reallocate(void* ptr1, const size_t n1, const size_t n2) {
  assert(ptr1 && n1 > 0u);
  assert(n2 > 0u);
#if !USE_MEMORY_POOL
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
    deallocate(ptr1, n1);
  }
  return ptr2;
#endif
}
