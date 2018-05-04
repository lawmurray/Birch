cpp {{
#include <chrono>

std::chrono::time_point<std::chrono::system_clock> savedTimePoint = std::chrono::system_clock::now();
}}

function tic() {
  cpp {{
  savedTimePoint = std::chrono::system_clock::now();
  }}
}

function toc() -> Real {
  elapsed:Real;
  cpp {{
  std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - savedTimePoint;
  elapsed_ = elapsed.count();
  }}
  return elapsed;
}
