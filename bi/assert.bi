import math;
import string;

hpp {{
#include <cassert>
}}

function require(condition:Boolean) {
  cpp {{
  assert(condition);
  }}
}
