cpp{{
#include <sys/time.h>
#include <sys/resource.h>
}}

/**
 * Get the Birch version number.
 */
function version() -> String {
  cpp{{
  return PACKAGE_VERSION;
  }}
}

/**
 * Get the CPU time used by the current process, in microseconds.
 *
 * The current implementation returns the sum of the user and system times
 * reported by `getrusage` (see `man getrusage`).
 */
function elapsed() -> Integer {
  cpp{{
  rusage usage;
  int ret = getrusage(RUSAGE_SELF, &usage);
  if (ret == 0) {
    auto user = usage.ru_utime;
    auto sys = usage.ru_stime;
    return bi::type::Integer(1e6)*(user.tv_sec + sys.tv_sec) + user.tv_usec +
        sys.tv_usec;
  } else {
    error("getrusage() call failed");
  }
  }}
}

/**
 * Get the peak memory use of the current process, in kilobytes.
 *
 * The current implementation returns the maximum resident set size reported
 * by `getrusage` (see `man getrusage`).
 */
function memory() -> Integer {
  cpp{{
  rusage usage;
  int ret = getrusage(RUSAGE_SELF, &usage);
  if (ret == 0) {
    return usage.ru_maxrss;
  } else {
    error("getrusage() call failed");
  }
  }}
}
