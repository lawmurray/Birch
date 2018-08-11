/**
 * @file
 */
#include "libbirch/global.hpp"

#include "libbirch/World.hpp"
#include "libbirch/Allocator.hpp"

#include <cstdio>
#include <cstdlib>

void* aligned_alloc(const size_t n) {
  void* ptr = nullptr;
  posix_memalign(&ptr, 64ull, n);
  assert(ptr);
  return ptr;
}

static bi::World* rootWorld = new bi::World(0);
bi::World* bi::fiberWorld = rootWorld;
bool bi::fiberClone = false;
std::vector<bi::StackFrame> bi::stacktrace(32);
// ^ reserving a non-zero size seems necessary to avoid segfault here
char* bi::smallBuffer = (char*)aligned_alloc(1ull << 34ull);
char* bi::largeBuffer = (char*)aligned_alloc(1ull << 34ull);
bi::Pool bi::pool[128];

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
  void* ptr = nullptr;
  if (n > 0ull) {
    /* bin the allocation */
    int i = bin(n);

    /* reuse allocation in the pool, or create a new one */
    ptr = pool[i].pop();
    if (!ptr) {
      size_t m = unbin(i);
      if (i <= 7) {
#pragma omp atomic capture
        {
          ptr = smallBuffer;
          smallBuffer += m;
        }
      } else {
#pragma omp atomic capture
        {
          ptr = largeBuffer;
          largeBuffer += m;
        }
      }
    }
    assert(ptr);
  }
  return ptr;
}

void* bi::reallocate(void* ptr1, const size_t n1, const size_t n2) {
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
}

void bi::deallocate(void* ptr, const size_t n) {
  if (n > 0ull) {
    assert(ptr);
    int i = bin(n);
    pool[i].push(ptr);
  }
}

void bi::deallocate(void* ptr, const unsigned n) {
  if (n > 0u) {
    assert(ptr);
    int i = bin(n);
    pool[i].push(ptr);
  }
}
