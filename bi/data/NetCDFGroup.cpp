/**
 * @file
 */
#include "bi/data/NetCDFGroup.hpp"

#include "bi/exception/NetCDFException.hpp"

#include <sstream>

bi::NetCDFGroup::NetCDFGroup(const char* path, const NetCDFFileMode mode) :
    mode(mode),
    own(true) {
  namespace fs = boost::filesystem;

  /* create temporary file if no path given */
  if (path == nullptr) {
    this->path = fs::temp_directory_path() / fs::unique_path();
    this->temp = true;
  } else {
    this->path = path;
    this->temp = false;
  }

  /* open file */
  const char* path1 = this->path.c_str();
  if (mode == NEW) {
    int status = ::nc_create(path1, NC_NETCDF4 | NC_NOCLOBBER, &ncid);
    if (status != NC_NOERR) {
      throw NetCDFException(path1, status);
    }
    status = ::nc_set_fill(ncid, NC_NOFILL, NULL);
    if (status != NC_NOERR) {
      throw NetCDFException(path1, status);
    }
  } else if (mode == REPLACE) {
    int status = ::nc_create(path1, NC_NETCDF4, &ncid);
    if (status != NC_NOERR) {
      throw NetCDFException(path1, status);
    }
    status = ::nc_set_fill(ncid, NC_NOFILL, NULL);
    if (status != NC_NOERR) {
      throw NetCDFException(path1, status);
    }
  } else if (mode == READ) {
    int status = ::nc_open(path1, NC_NOWRITE, &ncid);
    if (status != NC_NOERR) {
      throw NetCDFException(path1, status);
    }
  } else if (mode == WRITE) {
    int status = ::nc_open(path1, NC_WRITE, &ncid);
    if (status != NC_NOERR) {
      throw NetCDFException(path1, status);
    }
  }
}

bi::NetCDFGroup::NetCDFGroup(const int ncid, const NetCDFFileMode mode,
    const std::vector<size_t>& external) :
    ncid(ncid),
    mode(mode),
    own(false),
    temp(false),
    external(external) {
  //
}

bi::NetCDFGroup::NetCDFGroup(const NetCDFGroup& o) :
    ncid(o.ncid),
    mode(o.mode),
    path(o.path),
    own(false),
    temp(o.temp),
    external(o.external) {
  //
}

bi::NetCDFGroup::NetCDFGroup(NetCDFGroup&& o) :
    ncid(o.ncid),
    mode(o.mode),
    path(o.path),
    own(o.own),
    temp(o.temp),
    external(o.external) {
  o.own = false;
}

bi::NetCDFGroup::~NetCDFGroup() {
  namespace fs = boost::filesystem;

  if (own) {
    if (!temp) {
      ::nc_sync(ncid);
    }
    ::nc_close(ncid);
    if (temp) {
      fs::remove(path);
    }
  }
}

bi::NetCDFGroup bi::NetCDFGroup::createGroup(const char* name,
    const std::vector<size_t>& lengths) const {
  /* create group */
  int groupid, status;
  status = nc_def_grp(ncid, name, &groupid);
  if (status != NC_NOERR) {
    throw NetCDFException(path.string(), status);
  }

  /* combine dimension lengths */
  std::vector<size_t> all;
  all.insert(all.end(), lengths.begin(), lengths.end());
  all.insert(all.end(), external.begin(), external.end());

  return NetCDFGroup(groupid, mode, all);
}

bi::NetCDFGroup bi::NetCDFGroup::mapGroup(const char* name,
    const std::vector<size_t>& lengths) const {
  /* map group */
  int groupid, status;
  status = nc_inq_ncid(ncid, name, &groupid);
  if (status != NC_NOERR) {
    throw NetCDFException(path.string(), status);
  }

  /* combine dimension lengths */
  std::vector<size_t> all;
  all.insert(all.end(), lengths.begin(), lengths.end());
  all.insert(all.end(), external.begin(), external.end());

  return NetCDFGroup(groupid, mode, all);
}

template<class Type>
int bi::NetCDFGroup::createVar(const char* name,
    const std::vector<size_t>& lengths) {
  int status;

  /* combine dimension lengths */
  std::vector<size_t> all;
  all.insert(all.end(), lengths.begin(), lengths.end());
  all.insert(all.end(), external.begin(), external.end());

  /* define dimensions */
  std::vector<int> dimids(all.size());
  std::stringstream buf;
  for (int i = 0; i < all.size(); ++i) {
    buf.str("");
    buf << name << '_' << (i + 1) << '_';
    status = nc_def_dim(ncid, buf.str().c_str(), all[i], &dimids[i]);
    if (status != NC_NOERR) {
      throw NetCDFException(path.string(), status);
    }
  }

  /* define variable */
  int varid;
  status = nc_def_var(ncid, name, NetCDFValueType<Type>::value, all.size(),
      dimids.data(), &varid);
  if (status != NC_NOERR) {
    throw NetCDFException(path.string(), status);
  }

  return varid;
}

template int bi::NetCDFGroup::createVar<unsigned char>(const char* name,
    const std::vector<size_t>& lengths);
template int bi::NetCDFGroup::createVar<int64_t>(const char* name,
    const std::vector<size_t>& lengths);
template int bi::NetCDFGroup::createVar<int32_t>(const char* name,
    const std::vector<size_t>& lengths);
template int bi::NetCDFGroup::createVar<double>(const char* name,
    const std::vector<size_t>& lengths);
template int bi::NetCDFGroup::createVar<float>(const char* name,
    const std::vector<size_t>& lengths);

template<class Type>
int bi::NetCDFGroup::mapVar(const char* name,
    const std::vector<size_t>& lengths) {
  /* combine dimension lengths */
  std::vector<size_t> all;
  all.insert(all.end(), lengths.begin(), lengths.end());
  all.insert(all.end(), external.begin(), external.end());

  /* get variable id */
  int varid, status;
  status = nc_inq_varid(ncid, name, &varid);
  if (status != NC_NOERR) {
    throw NetCDFException(path.string(), status);
  }

  /* get information on variable */
  char name1[NC_MAX_NAME + 1];
  nc_type type1;
  int ndims1, natts1;
  int dimids1[NC_MAX_VAR_DIMS];
  status = nc_inq_var(ncid, varid, name1, &type1, &ndims1, dimids1, &natts1);
  if (status != NC_NOERR) {
    throw NetCDFException(path.string(), status);
  }

  /* check type */
  if (type1 != NetCDFValueType<Type>::value) {
    throw NetCDFException(path.string(),
        std::string() + "variable " + name + " has incorrect type");
  }
  ///@todo Consider allowing e.g. Real64 variables to take Real32 as input

  /* check number of dimensions */
  if (ndims1 != all.size()) {
    throw NetCDFException(path.string(),
        (std::stringstream() << "variable " << name << " has " << ndims1
            << " dimensions, should have " << all.size()).str());
  }

  /* check dimension lengths */
  size_t len1;
  for (int i = 0; i < all.size(); ++i) {
    status = ::nc_inq_dimlen(ncid, dimids1[i], &len1);
    if (status != NC_NOERR) {
      throw NetCDFException(path.string(), status);
    }
    if (len1 != all[i]) {
      throw NetCDFException(path.string(),
          (std::stringstream() << "Variable " << name << " dimension "
              << (i + 1) << " has length " << len1 << ", expected " << all[i] << '.').str());
    }
  }

  return varid;
}

template int bi::NetCDFGroup::mapVar<unsigned char>(const char* name,
    const std::vector<size_t>& lengths);
template int bi::NetCDFGroup::mapVar<int64_t>(const char* name,
    const std::vector<size_t>& lengths);
template int bi::NetCDFGroup::mapVar<int32_t>(const char* name,
    const std::vector<size_t>& lengths);
template int bi::NetCDFGroup::mapVar<double>(const char* name,
    const std::vector<size_t>& lengths);
template int bi::NetCDFGroup::mapVar<float>(const char* name,
    const std::vector<size_t>& lengths);
