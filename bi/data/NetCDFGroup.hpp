/**
 * @file
 */
#pragma once

#include "bi/data/NetCDFView.hpp"
#include "bi/data/PrimitiveValue.hpp"
#include "bi/data/Frame.hpp"
#include "bi/data/netcdf.hpp"

#include "boost/filesystem.hpp"

#include <vector>

namespace bi {
/**
 * Group for values in a NetCDF file.
 *
 * @ingroup library
 */
class NetCDFGroup {
public:
  /**
   * Construct as top-level NetCDF file.
   *
   * @param path File name.
   * @param mode File mode.
   */
  NetCDFGroup(const char* path = nullptr, const NetCDFFileMode mode = NEW);

  /**
   * Constructor.
   *
   * @param ncid NetCDF id.
   * @param external Lengths of external dimensions for all variables in this
   * group.
   */
  NetCDFGroup(const int ncid, const NetCDFFileMode mode = NEW,
      const std::vector<size_t>& external = std::vector<size_t>());

  /**
   * Copy constructor.
   */
  NetCDFGroup(const NetCDFGroup& o);

  /**
   * Move constructor.
   */
  NetCDFGroup(NetCDFGroup&& o);

  /**
   * Destructor.
   */
  ~NetCDFGroup();

  /**
   * Create new variable.
   *
   * @param value Value object.
   * @param frame Frame object.
   * @param name Name of the variable.
   */
  template<class Type, class Frame = EmptyFrame>
  void create(PrimitiveValue<Type,NetCDFGroup>& value, const Frame& frame =
      EmptyFrame(), const char* name = nullptr);

  /**
   * Release variable previously created.
   */
  template<class Type, class Frame = EmptyFrame>
  void release(PrimitiveValue<Type,NetCDFGroup>& value, const Frame& frame =
      EmptyFrame());

  /**
   * @name Groups.
   */
  //@{
  /**
   * Create group.
   */
  NetCDFGroup createGroup(const char* name,
      const std::vector<size_t>& lengths = std::vector<size_t>()) const;

  /**
   * Map group.
   */
  NetCDFGroup mapGroup(const char* name, const std::vector<size_t>& lengths =
      std::vector<size_t>()) const;
  //@}

  /**
   * @name Variables
   */
  //@{
  /**
   * Create variable.
   *
   * @tparam Type Scalar type.
   *
   * @param name Name of the variable.
   * @param lengths Dimension lengths.
   *
   * @return Variable id.
   */
  template<class Type>
  int createVar(const char* name, const std::vector<size_t>& lengths =
      std::vector<size_t>());

  /**
   * Map variable.
   *
   * @tparam Type Scalar type.
   *
   * @param name Name of the variable.
   * @param lengths Dimension lengths.
   *
   * @return Variable id.
   */
  template<class Type>
  int mapVar(const char* name, const std::vector<size_t>& lengths =
      std::vector<size_t>());
  //@}

  /**
   * NetCDF interface file id.
   */
  int ncid;

  /**
   * NetCDF file mode.
   */
  NetCDFFileMode mode;

  /**
   * Path of the file.
   */
  boost::filesystem::path path;

  /**
   * Do we own the file?
   */
  bool own;

  /**
   * Is it a temporary file?
   */
  bool temp;

  /**
   * Lengths of external dimensions for all variables in this group.
   */
  std::vector<size_t> external;
};
}

#include "bi/data/NetCDFPrimitiveValue.hpp"

#include <cassert>

template<class Type, class Frame>
void bi::NetCDFGroup::create(PrimitiveValue<Type,NetCDFGroup>& value,
    const Frame& frame, const char* name) {
  /* pre-condition */
  assert(name);

  std::fill(value.convolved.offsets.begin(), value.convolved.offsets.end(),
      0);
  frame.lengths(value.convolved.lengths.data());
  frame.strides(value.convolved.strides.data());

  if (mode == NEW || mode == REPLACE) {
    value.varid = createVar<Type>(name, value.convolved.lengths);
  } else {
    value.varid = mapVar<Type>(name, value.convolved.lengths);
  }
}

template<class Type, class Frame>
void bi::NetCDFGroup::release(PrimitiveValue<Type,NetCDFGroup>& value,
    const Frame& frame) {
  //
}
