/**
 * @file
 */
#include "libbirch/global.hpp"

#include "libbirch/World.hpp"
#include "libbirch/Allocator.hpp"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

/**
 * Allocate a large buffer for the heap. This attempts to determine
 * the amount of physical memory installed on the system and allocates a
 * buffer of virtual memory twice the size of this.
 */
static char* heap() {
  void* ptr = nullptr;
  size_t size = sysconf(_SC_PAGE_SIZE);
  size_t npages = sysconf(_SC_PHYS_PAGES);
  size_t n = 2u*npages*size;
  posix_memalign(&ptr, 64ull, n);
  assert(ptr);
  return (char*)ptr;
}

static bi::World* rootWorld = new bi::World(0);
bi::World* bi::fiberWorld = rootWorld;
bool bi::fiberClone = false;
std::vector<bi::StackFrame> bi::stacktrace(32);
// ^ reserving a non-zero size seems necessary to avoid segfault here

#if !DISABLE_POOL
char* bi::buffer = heap();
bi::Pool bi::pool[128];
#endif

void bi::abort() {
  abort("assertion failed");
}

void bi::unknown_option(const std::string& name) {
  printf("error: unknown option '%s'\n", name.c_str());
  std::string possible = name;
  std::replace(possible.begin(), possible.end(), '_', '-');
  if (name != possible) {
    printf("note: perhaps try '%s' instead?\n", possible.c_str());
  }
  std::exit(1);
}

void bi::abort(const std::string& msg) {
  printf("error: %s\n", msg.c_str());
#ifndef NDEBUG
  printf("stack trace:\n");
  unsigned i = 0;
  for (auto iter = stacktrace.rbegin(); i < 20u && iter != stacktrace.rend();
      ++iter) {
    printf("    %-24s @ %s:%d\n", iter->func, iter->file, iter->line);
    ++i;
  }
  if (i < stacktrace.size()) {
    int rem = stacktrace.size() - i;
    printf("  + %d more\n", rem);
  }
#endif
  std::exit(1);
}

void* bi::allocate(const size_t n) {
#if DISABLE_POOL
  return std::malloc(n);
#else
  void* ptr = nullptr;
  if (n > 0ull) {
    int i = bin(n);       // determine which pool
    ptr = pool[i].pop();  // attempt to reuse from this pool
    if (!ptr) {           // otherwise allocate new
      size_t m = unbin(i);
      #pragma omp atomic capture
      {
        ptr = buffer;
        buffer += m;
      }
    }
    assert(ptr);
  }
  return ptr;
#endif
}

void* bi::reallocate(void* ptr1, const size_t n1, const size_t n2) {
#if DISABLE_POOL
  return std::realloc(ptr1, n2);
#else
  int i1 = bin(n1);
  int i2 = bin(n2);
  void* ptr2 = ptr1;
  if (i1 != i2) {
    /* can't continue using current allocation */
    ptr2 = allocate(n2);
    std::memcpy(ptr2, ptr1, n1);
    deallocate(ptr1, n1);
  }
  return ptr2;
#endif
}

void bi::deallocate(void* ptr, const size_t n) {
#if DISABLE_POOL
  std::free(ptr);
#else
  if (n > 0ull) {
    assert(ptr);
    int i = bin(n);
    pool[i].push(ptr);
  }
#endif
}

void bi::deallocate(void* ptr, const unsigned n) {
#if DISABLE_POOL
  std::free(ptr);
#else
  if (n > 0u) {
    assert(ptr);
    int i = bin(n);
    pool[i].push(ptr);
  }
#endif
}
