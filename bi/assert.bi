import math;
import string;

hpp {{
#include "bi/misc/assert.hpp"
}}

function require(condition:Boolean) {
  cpp {{
  BI_ASSERT(condition);
  }}
}

function require(condition:Boolean, message:String) {
  cpp {{
  BI_ASSERT_MSG(condition, message);
  }}
}
