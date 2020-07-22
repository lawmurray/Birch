/**
 * Length of a matrix; equals `rows()`.
 */
function length<Value>(X:Value[_,_]) -> Integer {
  cpp{{
  return X.rows();
  }}
}

/**
 * Number of rows of a matrix.
 */
function rows<Value>(X:Value[_,_]) -> Integer {
  cpp{{
  return X.rows();
  }}
}

/**
 * Number of columns of a matrix.
 */
function columns<Value>(X:Value[_,_]) -> Integer {
  cpp{{
  return X.cols();
  }}
}

/**
 * Convert single-element matrix to scalar value.
 */
function scalar<Type>(X:Type[_,_]) -> Type {
  assert rows(X) == 1;  
  assert columns(X) == 1;  
  return X[1,1];
}

/**
 * Convert vector to matrix with single row.
 */
function row<Type>(x:Type[_]) -> Type[_,_] {
  return matrix(\(i:Integer, j:Integer) -> Type { return x[j]; }, 1,
      length(x));
}

/**
 * Convert vector to matrix with single column.
 */
function column<Type>(x:Type[_]) -> Type[_,_] {
  return matrix(\(i:Integer, j:Integer) -> Type {
        return x[i];
      }, 1, length(x));
}

/**
 * Vectorize a matrix by stacking its columns. This is the inverse operation
 * of `mat()`.
 *
 * - X: The matrix.
 */
function vec<Type>(X:Type[_,_]) -> Type[_] {
  auto R <- rows(X);
  auto C <- columns(X);
  return vector(\(i:Integer) -> Type {
        return X[mod(i - 1, R) + 1, (i - 1)/R + 1];
      }, R*C);
}

/**
 * Matrixize a vector by unstacking it into columns. This is the inverse
 * operation of `vec()`.
 *
 * - x: The vector.
 * - columns: The number of columns. Must be a factor of the length of `x`.
 */
function mat<Type>(x:Type[_], columns:Integer) -> Type[_,_] {
  assert mod(length(x), columns) == 0;
  auto R <- length(x)/columns;
  auto C <- columns;
  return matrix(\(i:Integer, j:Integer) -> Type {
        return x[(j - 1)*R + i];
      }, R, C);
}

/**
 * Stack two matrices atop one another (i.e. append columns) to create a
 * new matrix.
 */
function stack<Type>(X:Type[_,_], Y:Type[_,_]) -> Type[_,_] {
  assert columns(X) == columns(Y);
  
  auto R1 <- rows(X);
  auto R2 <- rows(Y);
  auto C <- columns(X);
  
  return matrix(\(i:Integer, j:Integer) -> Type {
        if i <= R1 {
          return X[i,j];
        } else {
          return Y[i - R1,j];
        }
      }, R1 + R2, C);
}

/**
 * Pack two matrices next to one another (i.e. append rows) to create a
 * new matrix.
 */
function pack<Type>(X:Type[_,_], Y:Type[_,_]) -> Type[_,_] {
  assert rows(X) == rows(Y);
  
  auto R <- rows(X);
  auto C1 <- columns(X);
  auto C2 <- columns(Y);
  
  return matrix(\(i:Integer, j:Integer) -> Type {
        if j <= C1 {
          return X[i,j];
        } else {
          return Y[i,j - C1];
        }
      }, R, C1 + C2);
}

/**
 * Create diagonal matrix, filling the diagonal with a given scalar value.
 *
 * - x: The value.
 * - length: Number of rows/columns.
 */
function diagonal(x:Real, length:Integer) -> Real[_,_] {
  return matrix(\(i:Integer, j:Integer) -> Real {
        if i == j {
          return x;
        } else {
          return 0.0;
        }
      }, length, length);
}

/**
 * Create diagonal matrix, filling the diagonal with a given scalar value.
 *
 * - x: The value.
 * - length: Number of rows/columns.
 */
function diagonal(x:Integer, length:Integer) -> Integer[_,_] {
  return matrix(\(i:Integer, j:Integer) -> Integer {
        if i == j {
          return x;
        } else {
          return 0;
        }
      }, length, length);
}

/**
 * Create diagonal matrix, filling the diagonal with a given scalar value.
 *
 * - x: The value.
 * - length: Number of rows/columns.
 */
function diagonal(x:Boolean, length:Integer) -> Boolean[_,_] {
  return matrix(\(i:Integer, j:Integer) -> Boolean {
        if i == j {
          return x;
        } else {
          return false;
        }
      }, length, length);
}


/**
 * Create identity matrix.
 *
 * - length: Number of rows/columns.
 */
function identity(length:Integer) -> Real[_,_] {
  return diagonal(1.0, length);
}

/**
 * Create a matrix filled by a lambda function.
 *
 * - λ: Lambda function.
 * - rows: Number of rows.
 * - columns: Number of columns.
 *
 * Returns: The new matrix.
 *
 * The lambda function is called once for each element in the new matrix,
 * receiving, as its argument, the row and column indices of that
 * element, and returning the value at that element.
 */
function matrix<Type>(λ:\(Integer, Integer) -> Type, rows:Integer,
    columns:Integer) -> Type[_,_] {
  cpp{{
  /* wrap λ in another lambda function to translate 0-based serial (row-wise)
   * indices into 1-based row/column indices */
  return libbirch::make_array_from_lambda<Type>(libbirch::make_shape(rows,
      columns), [&](int64_t i) { return λ(i/columns + 1, i%columns + 1); });
  }}
}

/**
 * Create matrix filled with a given scalar value.
 *
 * - x: The value.
 * - rows: Number of rows.
 * - columns: Number of columns.
 */
function matrix<Type>(x:Type, rows:Integer, columns:Integer) -> Type[_,_] {
  cpp{{
  return libbirch::make_array<Type>(libbirch::make_shape(rows, columns), x);
  }}
}

/**
 * Convert matrix to String.
 */
function String(X:Real[_,_]) -> String {
  result:String;
  cpp{{
  std::stringstream buf;
  }}
  for i in 1..rows(X) {
    cpp{{
    if (i > 1) {
      buf << '\n';
    }
    }}
    for j in 1..columns(X) {
      auto value <- X[i,j];
      cpp{{
      if (j > 1) {
        buf << ' ';
      }
      if (value == floor(value)) {
        buf << (int64_t)value << ".0";
      } else {
        buf << std::scientific << std::setprecision(14) << value;
      }
      }}
    }
  }
  cpp{{
  result = buf.str();
  }}
  return result;
}

/**
 * Convert matrix to String.
 */
function String(X:Integer[_,_]) -> String {
  result:String;
  cpp{{
  std::stringstream buf;
  }}
  for i in 1..rows(X) {
    cpp{{
    if (i > 1) {
      buf << '\n';
    }
    }}
    for j in 1..columns(X) {
      auto value <- X[i,j];
      cpp{{
      if (j > 1) {
        buf << ' ';
      }
      buf << value;
      }}
    }
  }
  cpp{{
  result = buf.str();
  }}
  return result;
}

/**
 * Convert matrix to String.
 */
function String(X:Boolean[_,_]) -> String {
  result:String;
  cpp{{
  std::stringstream buf;
  }}
  for i in 1..rows(X) {
    cpp{{
    if (i > 1) {
      buf << '\n';
    }
    }}
    for j in 1..columns(X) {
      auto value <- X[i,j];
      cpp{{
      if (j > 1) {
        buf << ' ';
      }
      if (value) {
        buf << "true";
      } else {
        buf << "false";
      }
      }}
    }
  }
  cpp{{
  result = buf.str();
  }}
  return result;
}

/**
 * Convert matrix to String.
 */
function String(X:LLT) -> String {
  return String(canonical(X));
}
