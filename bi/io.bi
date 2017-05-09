import basic;
import math.vector;

cpp {{
#include <cstdio>
}}

/**
 * Output to standard output.
 */
function print(value:String) {
  cpp {{
  ::printf("%s", value.c_str());
  }}
}

/**
 * Output to standard output.
 */
function print(value:Real) {
  cpp {{
  ::printf("%f", value);
  }}
}

/**
 * Output to standard output.
 */
function print(value:Integer) {
  cpp {{
  ::printf("%lld", value);
  }}
}

/**
 * Output to standard output.
 */
function print(value:Real[_]) {
  i:Integer;
  for (i in 1..length(value)) {
    if (i != 1) {
      print(", ");
    }
    print(value[i]);
  }
}
