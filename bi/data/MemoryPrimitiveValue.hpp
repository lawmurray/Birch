/**
 * @file
 */
#pragma once

#include "bi/data/MemoryGroup.hpp"

namespace bi {
/**
 * Value for MemoryGroup.
 *
 * @ingroup library
 *
 * @tparam Type Primitive type.
 */
template<class Type>
class PrimitiveValue<Type,MemoryGroup> {
public:
  typedef MemoryGroup group_type;
  typedef Type value_type;

  /**
   * Constructor.
   */
  template<class Tail, class Head>
  PrimitiveValue(const NonemptyFrame<Tail,Head>& frame, const char* name =
      nullptr, const MemoryGroup& group = MemoryGroup());

  /**
   * Constructor.
   */
  PrimitiveValue(const EmptyFrame& frame = EmptyFrame(), const char* name =
      nullptr, const MemoryGroup& group = MemoryGroup());

  /**
   * Value constructor.
   *
   * @internal Best to keep this explicit to avoid silently allocating
   * memory unintentionally.
   */
  explicit PrimitiveValue(const Type& value);

  /**
   * Copy constructor.
   */
  PrimitiveValue(const PrimitiveValue<Type,MemoryGroup>& o);

  /**
   * Move constructor.
   */
  PrimitiveValue(PrimitiveValue<Type,MemoryGroup> && o);

  /**
   * Copy constructor.
   */
  template<class Frame = EmptyFrame>
  PrimitiveValue(const PrimitiveValue<Type,MemoryGroup>& o, const bool deep =
      true, const Frame& frame = EmptyFrame(), const char* name = nullptr,
      const MemoryGroup& group = MemoryGroup());

  /**
   * View constructor.
   */
  template<class Frame, class View>
  PrimitiveValue(const PrimitiveValue<Type,MemoryGroup>& o,
      const Frame& frame, const View& view);

  /**
   * Destructor.
   */
  ~PrimitiveValue();

  /**
   * Copy assignment.
   */
  PrimitiveValue<Type,MemoryGroup>& operator=(
      const PrimitiveValue<Type,MemoryGroup>& o);

  /**
   * Move assignment.
   */
  PrimitiveValue<Type,MemoryGroup>& operator=(
      PrimitiveValue<Type,MemoryGroup> && o);

  /**
   * Generic assignment.
   */
  template<class Group1>
  PrimitiveValue<Type,MemoryGroup>& operator=(
      const PrimitiveValue<Type,Group1>& o);

  /**
   * Value assignment.
   */
  PrimitiveValue<Type,MemoryGroup>& operator=(const Type& o);

  /**
   * Call operator for lambda types.
   */
  template<class... Args>
  auto operator()(Args... args) const {
    return static_cast<const Type&>(*this)(args...);
  }

  /**
   * Value type conversion.
   */
  operator Type&();

  /**
   * Value type conversion.
   */
  operator const Type&() const;

  /**
   * Underlying buffer.
   */
  Type* ptr;

  /**
   * Group.
   */
  MemoryGroup group;

  /**
   * Do we own the underlying buffer?
   */
  bool own;
};
}

#include "bi/data/copy.hpp"

template<class Type>
template<class Tail, class Head>
bi::PrimitiveValue<Type,bi::MemoryGroup>::PrimitiveValue(
    const NonemptyFrame<Tail,Head>& frame, const char* name,
    const MemoryGroup& group) :
    group(group),
    own(true) {
  this->group.create(*this, frame, name);
}

template<class Type>
template<class Frame>
bi::PrimitiveValue<Type,bi::MemoryGroup>::PrimitiveValue(
    const PrimitiveValue<Type,MemoryGroup>& o, const bool deep,
    const Frame& frame, const char* name, const MemoryGroup& group) :
    group(group) {
  if (deep) {
    this->group.create(*this, frame, name);
    own = true;
    *this = o;
  } else {
    ptr = o.ptr;
    own = false;
  }
}

template<class Type>
template<class Frame, class View>
bi::PrimitiveValue<Type,bi::MemoryGroup>::PrimitiveValue(
    const PrimitiveValue<Type,MemoryGroup>& o, const Frame& frame,
    const View& view) :
    ptr(o.ptr + frame.serial(view)),
    group(o.group),
    own(false) {
  //
}

template<class Type>
template<class Group1>
bi::PrimitiveValue<Type,bi::MemoryGroup>& bi::PrimitiveValue<Type,
    bi::MemoryGroup>::operator=(const PrimitiveValue<Type,Group1>& o) {
  copy(*this, o);
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::PrimitiveValue(
    const EmptyFrame& frame, const char* name, const MemoryGroup& group) :
    group(group),
    own(true) {
  this->group.create(*this, frame, name);
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::PrimitiveValue(const Type& value) :
    own(true) {
  this->group.create(*this);
  this->group.fill(*this, value);
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::PrimitiveValue(
    const PrimitiveValue<Type,MemoryGroup>& o) :
    own(true) {
  this->group.create(*this);
  *this = o;
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::PrimitiveValue(
    PrimitiveValue<Type,MemoryGroup> && o) :
    ptr(o.ptr),
    group(o.group),
    own(o.own) {
  o.own = false;  // ownership moves
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::~PrimitiveValue() {
  if (own) {
    group.release(*this);
  }
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>& bi::PrimitiveValue<Type,
    bi::MemoryGroup>::operator=(const PrimitiveValue<Type,MemoryGroup>& o) {
  copy(*this, o);
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>& bi::PrimitiveValue<Type,
    bi::MemoryGroup>::operator=(PrimitiveValue<Type,MemoryGroup> && o) {
  if (ptr == o.ptr) {
    /* just take ownership */
    if (!own) {
      std::swap(own, o.own);
    }
  } else {
    /* copy assignment */
    own = true;
    this->group.create(*this);
    *this = o;
  }
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>& bi::PrimitiveValue<Type,
    bi::MemoryGroup>::operator=(const Type& o) {
  copy(*this, o);
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::operator Type&() {
  return *ptr;
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::operator const Type&() const {
  return *ptr;
}

extern template class bi::PrimitiveValue<unsigned char,bi::MemoryGroup>;
extern template class bi::PrimitiveValue<int64_t,bi::MemoryGroup>;
extern template class bi::PrimitiveValue<int32_t,bi::MemoryGroup>;
extern template class bi::PrimitiveValue<float,bi::MemoryGroup>;
extern template class bi::PrimitiveValue<double,bi::MemoryGroup>;
