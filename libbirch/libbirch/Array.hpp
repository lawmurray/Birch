/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/thread.hpp"
#include "libbirch/type.hpp"
#include "libbirch/Shape.hpp"
#include "libbirch/Iterator.hpp"
#include "libbirch/Eigen.hpp"
#include "libbirch/ReadersWriterLock.hpp"

namespace libbirch {
/**
 * Array.
 *
 * @tparam T Value type.
 * @tparam F Shape type.
 */
template<class T, class F>
class Array {
  template<class U, class G> friend class Array;
public:
  using this_type = Array<T,F>;
  using value_type = T;
  using shape_type = F;
  using eigen_type = typename eigen_type<this_type>::type;
  using eigen_stride_type = typename eigen_stride_type<this_type>::type;

  /**
   * Constructor.
   */
  Array() :
      shape(),
      buffer(nullptr),
      isView(false) {
    assert(shape.volume() == 0);
  }

  /**
   * Constructor.
   *
   * @param shape Shape.
   */
  Array(const F& shape) :
      shape(shape),
      buffer(nullptr),
      isView(false) {
    allocate();
    initialize();
  }

  /**
   * Constructor.
   *
   * @tparam ...Args Constructor parameter types.
   *
   * @param shape Shape.
   * @param args Constructor arguments.
   */
  template<class... Args, std::enable_if_t<std::is_constructible<T,Args...>::value,int> = 0>
  Array(const F& shape, Args&&... args) :
      shape(shape),
      buffer(nullptr),
      isView(false) {
    allocate();
    initialize(std::forward<Args>(args)...);
  }

  /**
   * Constructor.
   *
   * @param shape Shape.
   * @param values Values.
   */
  template<class G = F, std::enable_if_t<G::count() == 1,int> = 0>
  Array(const std::initializer_list<T>& values) :
      shape(values.size()),
      buffer(nullptr),
      isView(false) {
    allocate();
    std::uninitialized_copy(values.begin(), values.end(), begin());
  }

  /**
   * Constructor.
   *
   * @param shape Shape.
   * @param values Values.
   */
  template<class G = F, std::enable_if_t<G::count() == 2,int> = 0>
  Array(const std::initializer_list<std::initializer_list<T>>& values) :
      shape(values.size(), values.begin()->size()),
      buffer(nullptr),
      isView(false) {
    allocate();
    auto ptr = buf();
    for (auto row : values) {
      for (auto value : row) {
        new (ptr++) T(value);
      }
    }
  }

  /**
   * Constructor.
   *
   * @param l Lambda called to construct each element.
   * @param shape Shape.
   */
  template<class L>
  Array(const L& l, const F& shape) :
      shape(shape),
      buffer(nullptr),
      isView(false) {
    allocate();
    int64_t n = 0;
    for (auto iter = begin(); iter != end(); ++iter) {
      new (&*iter) T(l(n++));
    }
  }

  /**
   * Copy constructor.
   */
  Array(const Array& o) :
      shape(o.shape.compact()),
      buffer(nullptr),
      isView(false) {
    allocate();
    uninitialized_copy(o);
  }

  /**
   * Generic copy constructor.
   */
  template<class U, class G, std::enable_if_t<F::count() == G::count() &&
      std::is_convertible<U,T>::value,int> = 0>
  Array(const Array<U,G>& o) :
      shape(o.shape.compact()),
      buffer(nullptr),
      isView(false) {
    allocate();
    uninitialized_copy(o);
  }

  /**
   * Move constructor.
   */
  Array(Array&& o) : Array() {
    if (!o.isView) {
      swap(o);
    } else {
      shape = o.shape.compact();
      allocate();
      uninitialized_copy(o);
    }
  }

  /**
   * Destructor.
   */
  ~Array() {
    release();
  }

  /**
   * Copy assignment.
   */
  Array& operator=(const Array& o) {
    assign(o);
    return *this;
  }

  /**
   * Move assignment.
   */
  Array& operator=(Array&& o) {
    if (!isView && !o.isView) {
      swap(o);
    } else {
      assign(o);
    }
    return *this;
  }

  /**
   * Copy assignment. For a view the shapes of the two arrays must
   * conform, otherwise a resize is permitted.
   */
  void assign(const Array<T,F>& o) {
    if (isView) {
      libbirch_assert_msg_(o.shape.conforms(shape), "array sizes are different");
      copy(o);
    } else {
      Array<T,F> tmp(o);
      swap(tmp);
    }
  }

  /**
   * Number of elements.
   */
  auto size() const {
    return shape.size();
  }

  /**
   * Number of elements allocated.
   */
  auto volume() const {
    return shape.volume();
  }

  /**
   * @name Element access.
   */
  ///@{
  /**
   * Iterator pointing to the first element.
   *
   * Iterators are used to access the elements of an array sequentially.
   * Elements are visited in the order in which they are stored in memory;
   * the rightmost dimension is the fastest moving (for a matrix, this is
   * "row major" order).
   */
  Iterator<T,F> begin() {
    return Iterator<T,F>(buf(), shape);
  }
  Iterator<T,F> begin() const {
    return Iterator<T,F>(buf(), shape);
  }
  Iterator<T,F> end() {
    return begin() + size();
  }
  Iterator<T,F> end() const {
    return begin() + size();
  }

  /**
   * Slice.
   *
   * @tparam V Slice type.
   *
   * @param slice Slice.
   *
   * @return The resulting view or element.
   */
  template<class V, std::enable_if_t<V::rangeCount() != 0,int> = 0>
  auto slice(const V& slice) {
    return Array<T,decltype(shape(slice))>(shape(slice), buffer,
        shape.serial(slice));
  }
  template<class V, std::enable_if_t<V::rangeCount() != 0,int> = 0>
  auto slice(const V& slice) const {
    return Array<T,decltype(shape(slice))>(shape(slice), buffer,
        shape.serial(slice));
  }
  template<class V, std::enable_if_t<V::rangeCount() == 0,int> = 0>
  auto& slice(const V& slice) {
    return *(buf() + shape.serial(slice));
  }
  template<class V, std::enable_if_t<V::rangeCount() == 0,int> = 0>
  auto slice(const V& slice) const {
    return *(buf() + shape.serial(slice));
  }

  /**
   * Slice.
   *
   * @tparam ...Args Slice argument types.
   *
   * @param args... Slice arguments.
   *
   * @return The resulting view or element.
   */
  template<class... Args>
  decltype(auto) operator()(Args&&... args) {
    return slice(make_slice(std::forward<Args>(args)...));
  }
  template<class... Args>
  decltype(auto) operator()(Args&&... args) const {
    return slice(make_slice(std::forward<Args>(args)...));
  }
  ///@}

  /**
   * Compare.
   */
  template<class U, class G>
  bool operator==(const Array<U,G>& o) const {
    return std::equal(begin(), end(), o.begin());
  }
  template<class U, class G>
  bool operator!=(const Array<U,G>& o) const {
    return !(*this == o);
  }

  /**
   * @name Resize
   */
  ///@{
  /**
   * For a one-dimensional array, push an element onto the end. This increases
   * the array size by one.
   *
   * @param x Value.
   */
  void push(const T& x) {
    insert(size(), x);
  }

  /**
   * For a one-dimensional array, insert an element at a given position. This
   * increases the array size by one.
   *
   * @param i Position.
   * @param x Value.
   */
  void insert(const int64_t i, const T& x) {
    static_assert(F::count() == 1, "can only enlarge one-dimensional arrays");
    assert(!isView);

    auto n = size();
    auto s = F(n + 1);
    if (!buffer) {
      Array<T,F> tmp(s, x);
      swap(tmp);
    } else {
      buffer = (T*)std::realloc(buffer, s.volume()*sizeof(T));
      std::memmove((void*)(buf() + i + 1), (void*)(buf() + i), (n - i)*sizeof(T));
      new (buf() + i) T(x);
      shape = s;
    }
  }

  /**
   * For a one-dimensional array, erase elements from a given position. This
   * decreases the array size by the number of elements.
   *
   * @param i Position.
   * @param len Number of elements to erase.
   */
  void erase(const int64_t i, const int64_t len = 1) {
    static_assert(F::count() == 1, "can only shrink one-dimensional arrays");
    assert(!isView);
    assert(len > 0);
    assert(size() >= len);

    auto n = size();
    auto s = F(n - len);
    if (s.size() == 0) {
      release();
    } else {
      for (int j = i; j < i + len; ++j) {
        buf()[j].~T();
      }
      std::memmove((void*)(buf() + i), (void*)(buf() + i + len), (n - len - i)*sizeof(T));
      buffer = (T*)std::realloc(buffer, s.volume()*sizeof(T));
    }
    shape = s;
  }
  ///@}

  /**
   * @name Eigen integration
   */
  ///@{
  template<class Check = T, std::enable_if_t<std::is_arithmetic<Check>::value,int> = 0>
  operator eigen_type() const {
    return toEigen();
  }

  template<class Check = T, std::enable_if_t<std::is_arithmetic<Check>::value,int> = 0>
  auto toEigen() {
    return eigen_type(buf(), rows(), cols(), eigen_stride_type(rowStride(),
        colStride()));
  }

  template<class Check = T, std::enable_if_t<std::is_arithmetic<Check>::value,int> = 0>
  auto toEigen() const {
    return eigen_type(buf(), rows(), cols(), eigen_stride_type(rowStride(),
        colStride()));
  }

  /**
   * Construct from Eigen Matrix expression.
   */
  template<class EigenType, std::enable_if_t<is_eigen_compatible<this_type,EigenType>::value,int> = 0>
  Array(const Eigen::MatrixBase<EigenType>& o) :
      shape(o.rows(), o.cols()),
      buffer(nullptr),
      isView(false) {
    allocate();
    toEigen() = o;
  }

  /**
   * Construct from Eigen DiagonalWrapper expression.
   */
  template<class EigenType, std::enable_if_t<is_diagonal_compatible<this_type,EigenType>::value,int> = 0>
  Array(const Eigen::DiagonalWrapper<EigenType>& o) :
      shape(o.rows(), o.cols()),
      buffer(nullptr),
      isView(false) {
    allocate();
    toEigen() = o;
  }

  /**
   * Construct from Eigen TriangularView expression.
   */
  template<class EigenType, unsigned Mode, std::enable_if_t<is_triangle_compatible<this_type,EigenType>::value,int> = 0>
  Array(const Eigen::TriangularView<EigenType,Mode>& o) :
      shape(o.rows(), o.cols()),
      buffer(nullptr),
      isView(false) {
    allocate();
    toEigen() = o;
  }

  /**
   * Number of rows. For a vectoor, this is the length.
   */
  auto rows() const {
    assert(1 <= F::count() && F::count() <= 2);
    return shape.length(0);
  }

  /**
   * Number of columns. For a vector, this is 1.
   */
  auto cols() const {
    assert(1 <= F::count() && F::count() <= 2);
    return F::count() == 1 ? 1 : shape.length(1);
  }

  /**
   * Stride between rows.
   */
  auto rowStride() const {
    assert(1 <= F::count() && F::count() <= 2);
    return F::count() == 1 ? shape.volume() : shape.stride(0);
  }

  /**
   * Stride between columns.
   */
  auto colStride() const {
    assert(1 <= F::count() && F::count() <= 2);
    return F::count() == 1 ? shape.stride(0) : shape.stride(1);
  }
  ///@}

private:
  /**
   * Constructor for views.
   */
  Array(const F& shape, T* buffer, int64_t offset) :
      shape(shape),
      buffer(buffer + offset),
      isView(true) {
    //
  }

  /**
   * Raw pointer to underlying buffer.
   */
  T* buf() const {
    return buffer;
  }

  /**
   * Swap with another array.
   */
  void swap(Array<T,F>& o) {
    assert(!isView);
    assert(!o.isView);
    std::swap(shape, o.shape);
    std::swap(buffer, o.buffer);
  }

  /**
   * Allocate memory for array, leaving uninitialized.
   */
  void allocate() {
    assert(!buffer);
    size_t bytes = volume()*sizeof(T);
    if (bytes > 0) {
      buffer = (T*)std::malloc(bytes);
    }
  }

  /**
   * Deallocate memory of array.
   */
  void release() {
    if (!isView) {
      size_t bytes = volume()*sizeof(T);
      if (bytes > 0) {
        std::destroy(begin(), end());
        std::free(buffer);
        buffer = nullptr;
      }
    }
  }

  /**
   * Initialize allocated memory.
   *
   * @param args Constructor arguments.
   */
  template<class ... Args, std::enable_if_t<std::is_constructible<T,Args...>::value,int> = 0>
  void initialize(Args&&... args) {
    auto iter = begin();
    auto last = end();
    for (; iter != last; ++iter) {
      new (&*iter) T(std::forward<Args>(args)...);
    }
  }

  /**
   * Assign from another array.
   */
  template<class U>
  void copy(const U& o) {
    auto n = std::min(size(), o.size());
    auto begin1 = o.begin();
    auto end1 = begin1 + n;
    auto begin2 = begin();
    auto end2 = begin2 + n;
    if (inside(begin1, end1, begin2)) {
      std::copy_backward(begin1, end1, end2);
    } else {
      std::copy(begin1, end1, begin2);
    }
  }

  /**
   * Copy from another array.
   */
  template<class U>
  void uninitialized_copy(const U& o) {
    auto n = std::min(size(), o.size());
    auto begin1 = o.begin();
    auto end1 = begin1 + n;
    auto begin2 = begin();
    for (; begin1 != end1; ++begin1, ++begin2) {
      new (&*begin2) T(*begin1);
    }
  }

  /**
   * Shape.
   */
  F shape;

  /**
   * Buffer.
   */
  T* buffer;

  /**
   * Is this a view of another array? A view has stricter assignment
   * semantics, as it cannot be resized or moved.
   */
  bool isView;
};

/**
 * Default array for `D` dimensions.
 */
template<class T, int D>
using DefaultArray = Array<T,typename DefaultShape<D>::type>;

}
