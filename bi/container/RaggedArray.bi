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
    return length(values) == 0;
  }

  /**
   * Clear all elements.
   */
  function clear() {
    values:Type[_];
    offsets:Integer[_];
    ncols:Integer[_];
    
    this.values <- values;
    this.offsets <- offsets;
    this.ncols <- ncols;
    
    nrows <- 0;
    nelements <- 0;
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
   * `x`.
   */
  function set(i:Integer, x:Type[_]) {
    values[from(i)..to(i)] <- x;
  } 

  /**
   * Add a new row at the end.
   */
  function pushBack() {
    auto nrows <- this.nrows;
    auto nelements <- this.nelements;
    cpp{{    
    this_()->offsets.insert(nrows, nelements + 1);
    this_()->ncols.insert(nrows, 0);
    }}
    this.nrows <- nrows + 1;
  }

  /**
   * Add a new element to the end of the last row.
   */
  function pushBack(x:Type) {
    pushBack(nrows, x);
  }

  /**
   * Add a new element to the end of a specified row.
   *
   * - i: Row.
   * - x: Value.
   */
  function pushBack(i:Integer, x:Type) {
    auto j <- offsets[i] + ncols[i];
    cpp{{
    this_()->values.insert(j - 1, x);
    }}
    for k in (i + 1)..nrows {
      offsets[k] <- offsets[k] + 1;
    }
    ncols[i] <- ncols[i] + 1;
    nelements <- nelements + 1;
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
   * First serial index of a row.
   *
   * - i: Row.
   */
  function from(i:Integer) -> Integer {
    assert 1 <= i && i <= nrows;
    assert offsets[i] != 0;  // not an empty row
    return offsets[i];
  }
  
  /**
   * Last serial index of a row.
   *
   * - i: Row.
   */
  function to(i:Integer) -> Integer {
    assert 1 <= i && i <= nrows;
    assert offsets[i] != 0;  // not an empty row
    return offsets[i] + ncols[i] - 1;
  }
  
  /**
   * Serial index of row and column.
   *
   * - i: Row.
   * - j: Column.
   */
  function serial(i:Integer, j:Integer) -> Integer {
    assert 1 <= i && i <= nrows;
    assert 1 <= j && j <= ncols[i];
    return from(i) + j - 1;
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
