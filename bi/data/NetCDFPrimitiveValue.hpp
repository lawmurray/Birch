/**
 * @file
 */
#pragma once

#include "bi/data/NetCDFGroup.hpp"

namespace bi {
/**
 * Value for NetCDFGroup.
 *
 * @ingroup library
 *
 * @tparam Type Primitive type.
 */
template<class Type>
class PrimitiveValue<Type,NetCDFGroup> {
public:
  typedef NetCDFGroup group_type;

  /**
   * Constructor.
   */
  template<class Frame = EmptyFrame>
  PrimitiveValue(const Frame& frame = EmptyFrame(),
      const char* name = nullptr, const NetCDFGroup& group = NetCDFGroup());

  /**
   * View constructor.
   */
  template<class Frame, class View>
  PrimitiveValue(const PrimitiveValue<Type,NetCDFGroup>& o,
      const Frame& frame, const View& view);

//  /**
//   * Shallow copy constructor.
//   */
//  PrimitiveValue(const PrimitiveValue<Type,NetCDFGroup>& o);
//
//  /**
//   * Move constructor.
//   */
//  PrimitiveValue(PrimitiveValue<Type,NetCDFGroup> && o);

  /**
   * Destructor.
   */
  ~PrimitiveValue();

  /**
   * Assignment.
   */
  PrimitiveValue<Type,NetCDFGroup>& operator=(
      const PrimitiveValue<Type,NetCDFGroup>& o);

  /**
   * Generic assignment.
   */
  template<class Type1, class Group1>
  PrimitiveValue<Type,NetCDFGroup>& operator=(
      const PrimitiveValue<Type1,Group1>& o);

  /**
   * Value assignment.
   */
  PrimitiveValue<Type,NetCDFGroup>& operator=(const Type& o);

  /**
   * Basic type conversion.
   */
  operator Type() const;

  /**
   * NetCDF variable id.
   */
  int varid;

  /**
   * NetCDF convolved view.
   */
  NetCDFView convolved;

  /**
   * Group.
   */
  NetCDFGroup group;
};
}

#include "bi/data/copy.hpp"

template<class Type>
template<class Frame>
bi::PrimitiveValue<Type,bi::NetCDFGroup>::PrimitiveValue(const Frame& frame,
    const char* name, const NetCDFGroup& group) :
    convolved(frame),
    group(group) {
  this->group.create(*this, frame, name);
}

template<class Type>
template<class Frame, class View>
bi::PrimitiveValue<Type,bi::NetCDFGroup>::PrimitiveValue(
    const PrimitiveValue<Type,NetCDFGroup>& o, const Frame& frame,
    const View& view) :
    varid(o.varid),
    convolved(o.convolved),
    group(o.group) {
  convolved.convolve(view);
}

template<class Type>
template<class Type1, class Group1>
bi::PrimitiveValue<Type,bi::NetCDFGroup>& bi::PrimitiveValue<Type,
    bi::NetCDFGroup>::operator=(const PrimitiveValue<Type1,Group1>& o) {
  copy(*this, o);
  return *this;
}
