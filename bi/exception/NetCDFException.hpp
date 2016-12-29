/**
 * @file
 */
#pragma once

#include "bi/exception/Exception.hpp"

namespace bi {
/**
 * NetCDF exception.
 *
 * @ingroup compiler_exception
 */
struct NetCDFException: public Exception {
  /**
   * Constructor.
   *
   * @param status NetCDF status code.
   */
  NetCDFException(const int status);

  /**
   * Constructor.
   *
   * @param path NetCDF file path.
   * @param status NetCDF status code.
   */
  NetCDFException(const std::string& path, const int status);

  /**
   * Constructor.
   *
   * @param msg Custom error message.
   */
  NetCDFException(const std::string& path, const std::string& msg);
};
}
