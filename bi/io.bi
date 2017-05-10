import basic;
import math;

cpp {{
#include <cstdio>
}}

function print(value:Boolean) {
  if (value) {
    print("true");
  } else {
    print("false");
  }
}

function print(value:Real) {
  cpp {{
  ::printf("%f", value);
  }}
}

function print(value:Integer) {
  cpp {{
  ::printf("%lld", value);
  }}
}

function print(value:String) {
  cpp {{
  ::printf("%s", value.c_str());
  }}
}

function print(x:Real[_]) {
  i:Integer;
  for (i in 1..length(x)) {
    if (i != 1) {
      print(", ");
    }
    print(x[i]);
  }
}

function print(X:Real[_,_]) {
  i:Integer;
  j:Integer;
  for (i in 1..rows(X)) {
    for (j in 1..columns(X)) {
      if (j != 1) {
        print(", ");
      }
      print(X[i,j]);
    }
    print("\n");
  }
}
