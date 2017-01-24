/**
 * @file
 */
#include "bi/data/netcdf.hpp"

#include "bi/exception/NetCDFException.hpp"

#include <cassert>

void bi::get(int ncid, int varid, const size_t start[], unsigned char* tp) {
  int status = nc_get_var1_uchar(ncid, varid, start, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::get(int ncid, int varid, const size_t start[], float* tp) {
  int status = nc_get_var1_float(ncid, varid, start, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::get(int ncid, int varid, const size_t start[], double* tp) {
  int status = nc_get_var1_double(ncid, varid, start, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::get(int ncid, int varid, const size_t start[], int32_t* tp) {
  int status = nc_get_var1_int(ncid, varid, start, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::get(int ncid, int varid, const size_t start[], int64_t* tp) {
  int status;
  if (sizeof(long) == 8) {
    status = nc_get_var1_long(ncid, varid, start, (long*)tp);
  } else if (sizeof(long long) == 8) {
    status = nc_get_var1_longlong(ncid, varid, start, (long long*)tp);
  } else {
    assert(false);
  }
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::get(int ncid, int varid, const size_t start[], const char** tp) {
  assert(false);
}

void bi::get(int ncid, int varid, const size_t start[],
    std::function<void()>* tp) {
  assert(false);
}

void bi::get(int ncid, int varid, const size_t start[], const size_t count[],
    unsigned char* tp) {
  int status = nc_get_vara_uchar(ncid, varid, start, count, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::get(int ncid, int varid, const size_t start[], const size_t count[],
    float* tp) {
  int status = nc_get_vara_float(ncid, varid, start, count, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::get(int ncid, int varid, const size_t start[], const size_t count[],
    double* tp) {
  int status = nc_get_vara_double(ncid, varid, start, count, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::get(int ncid, int varid, const size_t start[], const size_t count[],
    int32_t* tp) {
  int status = nc_get_vara_int(ncid, varid, start, count, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::get(int ncid, int varid, const size_t start[], const size_t count[],
    int64_t* tp) {
  int status;
  if (sizeof(long) == 8) {
    status = nc_get_vara_long(ncid, varid, start, count, (long*)tp);
  } else if (sizeof(long long) == 8) {
    status = nc_get_vara_longlong(ncid, varid, start, count, (long long*)tp);
  } else {
    assert(false);
  }
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::get(int ncid, int varid, const size_t start[], const size_t count[],
    const char** tp) {
  assert(false);
}

void bi::get(int ncid, int varid, const size_t start[], const size_t count[],
    std::function<void()>* tp) {
  assert(false);
}

void bi::put(int ncid, int varid, const size_t start[],
    const unsigned char* tp) {
  int status = nc_put_var1_uchar(ncid, varid, start, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::put(int ncid, int varid, const size_t start[], const float* tp) {
  int status = nc_put_var1_float(ncid, varid, start, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::put(int ncid, int varid, const size_t start[], const double* tp) {
  int status = nc_put_var1_double(ncid, varid, start, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::put(int ncid, int varid, const size_t start[], const int32_t* tp) {
  int status = nc_put_var1_int(ncid, varid, start, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::put(int ncid, int varid, const size_t start[], const int64_t* tp) {
  int status;
  if (sizeof(long) == 8) {
    status = nc_put_var1_long(ncid, varid, start, (long*)tp);
  } else if (sizeof(long long) == 8) {
    status = nc_put_var1_longlong(ncid, varid, start, (long long*)tp);
  } else {
    assert(false);
  }
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::put(int ncid, int varid, const size_t start[], const char** tp) {
  assert(false);
}

void bi::put(int ncid, int varid, const size_t start[],
    const std::function<void()>* tp) {
  assert(false);
}

void bi::put(int ncid, int varid, const size_t start[], const size_t count[],
    const unsigned char* tp) {
  int status = nc_put_vara_uchar(ncid, varid, start, count, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::put(int ncid, int varid, const size_t start[], const size_t count[],
    const float* tp) {
  int status = nc_put_vara_float(ncid, varid, start, count, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::put(int ncid, int varid, const size_t start[], const size_t count[],
    const double* tp) {
  int status = nc_put_vara_double(ncid, varid, start, count, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::put(int ncid, int varid, const size_t start[], const size_t count[],
    const int32_t* tp) {
  int status = nc_put_vara_int(ncid, varid, start, count, tp);
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::put(int ncid, int varid, const size_t start[], const size_t count[],
    const int64_t* tp) {
  int status;
  if (sizeof(long) == 8) {
    status = nc_put_vara_long(ncid, varid, start, count, (long*)tp);
  } else if (sizeof(long long) == 8) {
    status = nc_put_vara_longlong(ncid, varid, start, count, (long long*)tp);
  } else {
    assert(false);
  }
  if (status != NC_NOERR) {
    throw NetCDFException(status);
  }
}

void bi::put(int ncid, int varid, const size_t start[], const size_t count[],
    const char** tp) {
  assert(false);
}

void bi::put(int ncid, int varid, const size_t start[], const size_t count[],
    const std::function<void()>* tp) {
  assert(false);
}
