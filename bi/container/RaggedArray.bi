/**
 * Two-dimensional array where each row has a varying length. Internally, this
 * is stored in one contiguous array for $O(1)$ random access.
 */
final class RaggedArray<Type> {
  /**
   * Elements.
   */
  values:Type[_];

  /**
   * Offset into `values` for each row.
   */
  offsets:Integer[_];
  
  /**
   * Number of columns in each row.
   */
  ncols:Integer[_];
  
  /**
   * Number of rows.
   */
  nrows:Integer <- 0;
  
  /**
   * Number of elements.
   */
  nelements:Integer <- 0;

  /**
   * Is this empty?
   */
  function empty() -> Boolean {
    return nelements == 0;
  }

  /**
   * Clear all elements.
   */
  function clear() {
    shrink(0);
  }

  /**
   * Number of rows.
   */
  function size() -> Integer {
    return nrows;
  }
  
  /**
   * Number of elements for a given row.
   *
   * - i: Row.
   */
  function size(i:Integer) -> Integer {
    assert 0 < i && i <= nrows;
    return ncols[i];
  }
  
  /**
   * Get an element.
   *
   * - i: Row.
   * - j: Column.
   */
  function get(i:Integer, j:Integer) -> Type {
    return values[serial(i, j)];
  }

  /**
   * Get a row.
   *
   * - i: Row.
   */
  function get(i:Integer) -> Type[_] {
    assert 0 < i && i <= nrows;
    return values[from(i)..to(i)];
  }

  /**
   * Set an element.
   *
   * - i: Row.
   * - j: Column.
   * - x: Value.
   */
  function set(i:Integer, j:Integer, x:Type) {
    values[serial(i, j)] <- x;
  }

  /**
   * Set a row.
   *
   * - i: Row.
   * - x: Values.
   *
   * The number of columns in the row must match the number of columns in
   * `x`. Use `shrink` or `enlarge` beforehand if necessary.
   */
  function set(i:Integer, x:Type[_]) {
    values[from(i)..to(i)] <- x;
  } 

  /**
   * Add a new row at the back.
   */
  function pushBack() {
    enlarge(nrows + 1);
  }

  /**
   * Add a new element to the end of a row.
   *
   * - i: Row.
   * - x: Value.
   */
  function pushBack(i:Integer, x:Type) {
    enlarge(i, ncols[i] + 1, x);
  }

  /**
   * Iterate over the rows.
   *
   * Return: a fiber object that yields each row in forward order.
   */
  fiber walk() -> Type[_] {
    for i in 1..nrows {
      yield get(i);
    }
  }

  /**
   * Iterate over the columns of a row.
   *
   * - i: Row.
   *
   * Return: a fiber object that yields each row in forward order.
   */
  fiber walk(i:Integer) -> Type {
    for j in 1..ncols[i] {
      yield get(i, j);
    }
  }

  /**
   * Decrease the number of rows.
   *
   * - n: Number of rows.
   *
   * The current contents is preserved.
   */
  function shrink(n:Integer) {
    assert n < nrows;

    if (n == 0) {
      nelements <- 0;
    } else {
      nelements <- offsets[n] + ncols[n];
    }
    nrows <- n;
    cpp{{
    self_()->offsets.shrink(libbirch::make_shape(n));
    self_()->ncols.shrink(libbirch::make_shape(n));
    self_()->values.shrink(libbirch::make_shape(self_()->nelements));
    }}
    
    assert length(offsets) == length(ncols);
  }

  /**
   * Decrease the number of columns in a row.
   *
   * - i: Row.
   * - n: Number of columns.
   *
   * The current contents of the row is preserved.
   */
  function shrink(i:Integer, n:Integer) {
    assert n < ncols[i];
  
    d:Integer <- ncols[i] - n;
    if offsets[i] + ncols[i] - 1 < nelements {
      values[(offsets[i] + ncols[i] - d)..(nelements - d)] <- values[(offsets[i] + ncols[i])..nelements];
    }
    cpp{{
    self_()->values.shrink(libbirch::make_shape(self_()->nelements - d));
    }}
    ncols[i] <- n;
    if i < nrows {
      offsets[(i + 1)..nrows] <- offsets[(i + 1)..nrows] - d;
    }
    nelements <- nelements - d;
  }

  /**
   * Increase the number of rows.
   *
   * - n: Number of rows.
   *
   * The current contents is preserved.
   */
  function enlarge(n:Integer) {
    assert n > nrows;

    nrows <- n;
    cpp{{    
    self_()->offsets.enlarge(libbirch::make_shape(n), self_()->nelements + 1);
    self_()->ncols.enlarge(libbirch::make_shape(n), 0);
    }}

    assert length(offsets) == length(ncols);
  }
  
  /**
   * Increase the number of columns in a row.
   *
   * - i: Row.
   * - n: Number of columns.
   * - x: Value for new elements.
   *
   * The current contents of the row is preserved.
   */
  function enlarge(i:Integer, n:Integer, x:Type) {
    assert n > ncols[i];
  
    d:Integer <- n - ncols[i];
    cpp{{
    self_()->values.enlarge(libbirch::make_shape(self_()->nelements + d), x);
    }}
    if offsets[i] + ncols[i] - 1 < nelements {
      values[(offsets[i] + ncols[i] + d)..(nelements + d)] <- values[(offsets[i] + ncols[i])..nelements];
      for j in (offsets[i] + ncols[i])..(offsets[i] + ncols[i] + d - 1) {
        values[j] <- x;
      }
    }
    ncols[i] <- n;
    if i < nrows {
      offsets[(i + 1)..nrows] <- offsets[(i + 1)..nrows] + d;
    }
    nelements <- nelements + d;
  }

  /**
   * First serial index of a row.
   *
   * - i: Row.
   */
  function from(i:Integer) -> Integer {
    assert 0 < i && i <= nrows;
    return offsets[i];
  }
  
  /**
   * Last serial index of a row.
   *
   * - i: Row.
   */
  function to(i:Integer) -> Integer {
    assert 0 < i && i <= nrows;
    return offsets[i] + ncols[i] - 1;
  }
  
  /**
   * Serial index of row and column.
   *
   * - i: Row.
   * - j: Column.
   */
  function serial(i:Integer, j:Integer) -> Integer {
    assert 0 < i && i <= nrows;
    assert 0 < j && j <= ncols[i];
    return offsets[i] + j - 1;
  }

 function read(buffer:Buffer) {
    auto row <- buffer.walk();
    while row? {
      pushBack();
      auto col <- row!.walk();
      while col? {
        /* tricky, but works for both value and class types */
        auto x <- make<Type>();
        auto y <- col!.get(x);
        if y? {
          x <- Type?(y);  // cast needed for y:Object?
          pushBack(size(), x!);
        }
      }
    }
  }

  function write(buffer:Buffer) {
    buffer.setArray();
    for i in 1..size() {
      auto row <- buffer.push();
      row.setArray();
      for j in 1..size(i) {
        row.push().set(get(i, j));
      }
    }
  }
}
