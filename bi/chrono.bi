cpp{{
#include <chrono>

extern std::chrono::time_point<std::chrono::system_clock> savedTimePoint;
#pragma omp threadprivate(savedTimePoint)
std::chrono::time_point<std::chrono::system_clock> savedTimePoint = std::chrono::system_clock::now();
}}

/**
 * Reset timer.
 */
function tic() {
  cpp {{
  savedTimePoint = std::chrono::system_clock::now();
  }}
}

/**
 * Number of seconds since last call to `tic()`.
 */
function toc() -> Real {
  elapsed:Real;
  cpp {{
  std::chrono::duration<double> e = std::chrono::system_clock::now() - savedTimePoint;
  elapsed = e.count();
  }}
  return elapsed;
}
