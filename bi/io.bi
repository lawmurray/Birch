import basic;
import math;

cpp {{
#include <iostream>
#include <fstream>
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
  std::cout << value_;
  }}
}

function print(value:Integer) {
  cpp {{
  std::cout << value_;
  }}
}

function print(value:String) {
  cpp {{
  std::cout << value_;
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

function print(x:Integer[_]) {
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

function print(X:Integer[_,_]) {
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

/**
 * Read numbers from a file.
 */
function read(file:String, N:Integer) -> Real[_] {
  cpp{{
  std::ifstream stream(file_);
  }}
  x:Real[N];
  n:Integer;
  v:Real;
  for (n in 1..N) {
    cpp{{
    stream >> v_;
    }}
    x[n] <- v;
  }
  return x;
}
