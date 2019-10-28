/**
 * @file
 *
 * Definition of all global variables in one file, to ensure correct
 * initialization order.
 */
#include "libbirch/memory.hpp"
#include "libbirch/thread.hpp"

/* declared in memory.hpp */
libbirch::Atomic<size_t> libbirch::memoryUse(0);

#if ENABLE_MEMORY_POOL
/**
 * Allocate a large buffer for the heap.
 */
static char* heap() {
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

  libbirch::bufferStart = (char*)ptr;
  libbirch::bufferSize = n;

  return (char*)ptr;
}

libbirch::Atomic<char*> libbirch::buffer(heap());
char* libbirch::bufferStart;
size_t libbirch::bufferSize;
#endif

/* declared in thread.hpp */
static libbirch::Label* root() {
  static libbirch::SharedPtr<libbirch::Label> context(new libbirch::Label());
  return context.get();
}

libbirch::Label* libbirch::rootContext = root();
libbirch::EntryExitLock libbirch::freezeLock;
libbirch::EntryExitLock libbirch::finishLock;
