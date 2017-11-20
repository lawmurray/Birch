/**
 * An implementation of atomic_allocator, based on the implementation of
 * traceable_allocator in gc/gc_allocator.h of the Boehm garbage collector.
 */
template<class Type>
class atomic_allocator {
public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef Type* pointer;
  typedef const Type* const_pointer;
  typedef Type& reference;
  typedef const Type& const_reference;
  typedef Type value_type;

  template<class Type1> struct rebind {
    typedef atomic_allocator<Type1> other;
  };

  pointer address(reference value) const {
    return &value;
  }
  const_pointer address(const_reference value) const {
    return &value;
  }

  Type* allocate(size_type n, const void* = nullptr) {
    return static_cast<Type*>(GC_MALLOC_ATOMIC(n*sizeof(Type)));
  }

  void deallocate(pointer ptr, size_type n) {
    //GC_FREE(ptr);
  }

  size_type max_size() const throw () {
    return size_t(-1) / sizeof(Type);
  }

  void construct(pointer ptr, const Type& value) {
    new (ptr) Type(value);
  }

  void destroy(pointer ptr) {
    ptr->~Type();
  }
};

template<>
class atomic_allocator<void> {
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef void* pointer;
  typedef const void* const_pointer;
  typedef void value_type;

  template<class Type1> struct rebind {
    typedef atomic_allocator<Type1> other;
  };
};
