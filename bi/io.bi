import string;
import math;

cpp {{
#include <cstdio>
}}

/**
 * Output to standard output.
 */
function print(value:String) {
  cpp {{
  ::printf("%s", (const char*)(value));
  }}
}

/**
 * Output to standard output.
 */
function print(value:Real) {
  cpp {{
  ::printf("%f", double(value));
  }}
}

/**
 * Output to standard output.
 */
function print(value:Integer) {
  cpp {{
  ::printf("%lld", int64_t(value));
  }}
}
