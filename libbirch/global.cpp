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

bi::World* bi::fiberWorld = new bi::World(0);
bool bi::fiberClone = false;
std::list<bi::StackFrame> bi::stacktrace;
char* bi::smallBuffer = (char*)aligned_alloc(1ull << 36ull);
char* bi::largeBuffer = (char*)aligned_alloc(1ull << 36ull);
std::stack<void*,std::vector<void*>> bi::pool[128];

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
  for (auto iter = stacktrace.begin(); i < 20u && iter != stacktrace.end();
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
    auto& p = pool[i];
    if (p.empty()) {
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
    } else {
      ptr = p.top();
      p.pop();
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
