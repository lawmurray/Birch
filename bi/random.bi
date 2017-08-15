import math;

hpp {{
extern std::mt19937_64 rng;
}}

cpp {{
std::mt19937_64 rng(std::time(0));
}}

/**
 * Seed the pseudorandom number generator.
 *
 * `seed` Seed.
 */
function seed(s:Integer) {
  cpp {{
  rng.seed(s_);
  }}
}
