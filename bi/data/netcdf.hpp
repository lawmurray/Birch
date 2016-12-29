/**
 * @file
 */
#pragma once

#include <netcdf.h>
#include <cstdlib>

namespace bi {
/**
 * NetCDF file modes.
 *
 * @internal
 */
enum NetCDFFileMode {
  READ, WRITE, NEW, REPLACE
};

/**
 * Mappings from C++ to NetCDF data types.
 *
 * @internal
 */
template<class T>
struct NetCDFValueType {
  //
};
template<>
struct NetCDFValueType<unsigned char> {
  static constexpr nc_type value = NC_UBYTE;
};
template<>
struct NetCDFValueType<float> {
  static constexpr nc_type value = NC_FLOAT;
};
template<>
struct NetCDFValueType<double> {
  static constexpr nc_type value = NC_DOUBLE;
};
template<>
struct NetCDFValueType<int32_t> {
  static constexpr nc_type value = NC_INT;
};
template<>
struct NetCDFValueType<int64_t> {
  static constexpr nc_type value = NC_INT64;
};
template<>
struct NetCDFValueType<char*> {
  static constexpr nc_type value = NC_STRING;
};

/**
 * @name NetCDF read functions.
 */
//@{
void get(int ncid, int varid, const size_t start[], unsigned char* tp);
void get(int ncid, int varid, const size_t start[], float* tp);
void get(int ncid, int varid, const size_t start[], double* tp);
void get(int ncid, int varid, const size_t start[], int32_t* tp);
void get(int ncid, int varid, const size_t start[], int64_t* tp);
void get(int ncid, int varid, const size_t start[], const char** tp);

void get(int ncid, int varid, const size_t start[], const size_t count[],
    unsigned char* tp);
void get(int ncid, int varid, const size_t start[], const size_t count[],
    float* tp);
void get(int ncid, int varid, const size_t start[], const size_t count[],
    double* tp);
void get(int ncid, int varid, const size_t start[], const size_t count[],
    int32_t* tp);
void get(int ncid, int varid, const size_t start[], const size_t count[],
    int64_t* tp);
void get(int ncid, int varid, const size_t start[], const size_t count[],
    const char** tp);
//@}

/**
 * @name NetCDF write functions.
 */
//@{
void put(int ncid, int varid, const size_t start[], const unsigned char* tp);
void put(int ncid, int varid, const size_t start[], const float* tp);
void put(int ncid, int varid, const size_t start[], const double* tp);
void put(int ncid, int varid, const size_t start[], const int32_t* tp);
void put(int ncid, int varid, const size_t start[], const int64_t* tp);
void put(int ncid, int varid, const size_t start[], const char** tp);

void put(int ncid, int varid, const size_t start[], const size_t count[],
    const unsigned char* tp);
void put(int ncid, int varid, const size_t start[], const size_t count[],
    const float* tp);
void put(int ncid, int varid, const size_t start[], const size_t count[],
    const double* tp);
void put(int ncid, int varid, const size_t start[], const size_t count[],
    const int32_t* tp);
void put(int ncid, int varid, const size_t start[], const size_t count[],
    const int64_t* tp);
void put(int ncid, int varid, const size_t start[], const size_t count[],
    const char** tp);
//@}
}
