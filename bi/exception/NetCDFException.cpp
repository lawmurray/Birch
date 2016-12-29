/**
 * @file
 */
#include "bi/exception/NetCDFException.hpp"

#include "bi/data/netcdf.hpp"

#include <sstream>

bi::NetCDFException::NetCDFException(const int status) {
  std::stringstream base;
  base << "error: NetCDF error '" << nc_strerror(status) << "'\n";
  msg = base.str();
}

bi::NetCDFException::NetCDFException(const std::string& path,
    const int status) {
  std::stringstream base;
  base << "error: NetCDF error '" << nc_strerror(status) << "' for file '" << path << "'\n";
  msg = base.str();
}

bi::NetCDFException::NetCDFException(const std::string& path,
    const std::string& msg) {
  std::stringstream base;
  base << "error: NetCDF error '" << msg << "' for file '" << path << "'\n";
  this->msg = base.str();
}
