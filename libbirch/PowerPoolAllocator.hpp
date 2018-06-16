/**
 * @file
 */
#pragma once

#include <vector>
#include <stack>

#if __cplusplus > 201703L
#include <new>
#endif

namespace bi {
/**
 * Allocation pool.
 */
extern std::stack<void*> pool[];

/**
 * Heap allocator for objects of class type. Maintains a pool of allocations,
 * rounding sizes up to the nearest power of two for efficient reuse.
 *
 * @ingroup libbirch
 *
 * @tparam T Value type.
 */
template<class T>
class PowerPoolAllocator {
public:
  using value_type = T;

  PowerPoolAllocator() = default;
  PowerPoolAllocator(const PowerPoolAllocator<T>&) = default;
  PowerPoolAllocator(PowerPoolAllocator<T> &&) = default;

  template<class U>
  PowerPoolAllocator(const PowerPoolAllocator<U>& o) {
    //
  }

  static value_type* allocate(const size_t n) {
    value_type* ptr = nullptr;
    if (n > 0) {
      /* bin the allocation */
      int i = bin(n);

      /* reuse allocation in the pool, or create a new one */
      if (pool[i].empty()) {
        ptr = static_cast<value_type*>(std::malloc(1 << i));
      } else {
        auto& p = pool[i];
        ptr = static_cast<value_type*>(p.top());
        p.pop();
      }
      assert(ptr);
    }
    return ptr;
  }

  static value_type* reallocate(value_type* ptr1, const size_t n1,
      const size_t n2) {
    value_type* ptr2 = nullptr;

    /* bin the current allocation */
    int i1 = bin(n1);

    /* bin the new allocation */
    int i2 = bin(n2);

    if (n1 > 0 && i1 == i2) {
      /* current allocation is large enough, reuse */
      ptr2 = ptr1;
    } else {
      /* return the current allocation to the pool */
      if (n1 > 0) {
        pool[i1].push(ptr1);
      }

      if (n2 > 0) {
        /* reuse allocation in the pool, or create a new one */
        if (pool[i2].empty()) {
          ptr2 = static_cast<value_type*>(std::malloc(1 << i2));
        } else {
          auto& p = pool[i2];
          ptr2 = static_cast<value_type*>(p.top());
          p.pop();
        }
        assert(ptr2);

        /* copy over contents */
        std::memcpy(ptr2, ptr1, n1 * sizeof(T));
      }
    }
    return ptr2;
  }

  static void deallocate(value_type* ptr, const size_t n) {
    if (n > 0) {
      assert(ptr);

      /* bin the allocation */
      int i = bin(n);

      /* return this allocation to the pool */
      pool[i].push(ptr);
    }
  }

  /**
   * Determine in which bin an allocation of size @p n belongs. Return the
   * index of the bin and the size of allocations in that bin (which will
   * be greater than or equal to @p n).
   */
  static int bin(const size_t n) {
    /* minimum allocation size */
    #if __cplusplus > 201703L
    static const size_t minSize = std::hardware_destructive_interference_size;
    #else
    static const size_t minSize = 1u;
    #endif

    size_t m = std::max(n * sizeof(T), minSize) - 1;
    #if __has_builtin(__builtin_clzl)
    return sizeof(unsigned long)*8 - __builtin_clzl(m);
    #else
    int ret = 1;
    while ((m >> ret) > 0) {
      ++ret;
    }
    return ret;
    #endif
  }

  bool operator==(const PowerPoolAllocator<T>& o) const {
    return true;
  }

  bool operator!=(const PowerPoolAllocator<T>& o) const {
    return false;
  }
};
}
