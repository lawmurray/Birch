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
  sizes:Integer[_];
  
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
    sizes:Integer[_];
    
    this.values <- values;
    this.offsets <- offsets;
    this.sizes <- sizes;
  }

  /**
   * Number of rows.
   */
  function size() -> Integer {
    return length(offsets);
  }
  
  /**
   * Number of elements for a given row.
   *
   * - i: Row.
   */
  function size(i:Integer) -> Integer {
    return sizes[i];
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
   * Remove the first row.
   */
  function popFront() {
    assert size() > 0;
    if size() == 1 {
      clear();
    } else {
      auto ncols <- sizes[1];
      cpp{{
      this_()->offsets.erase(0);
      this_()->sizes.erase(0);
      }}
      if ncols > 0 {
        cpp{{
        this_()->values.erase(0, ncols);
        }}
        for k in 1..length(offsets) {
          offsets[k] <- offsets[k] - ncols;
        }
      }
    }
  }

  /**
   * Remove the first element from a specified row.
   *
   * - i: Row.
   */
  function popFront(i:Integer) {
    assert size(i) > 0;
    auto j <- offsets[i];
    cpp{{
    this_()->values.erase(j - 1);
    }}
    for k in (i + 1)..length(offsets) {
      offsets[k] <- offsets[k] - 1;
    }
    sizes[i] <- sizes[i] - 1;
  }

  /**
   * Add a new row at the end.
   */
  function pushBack() {
    auto nrows <- length(offsets);
    auto nvalues <- length(values);
    cpp{{    
    this_()->offsets.insert(nrows, nvalues + 1);
    this_()->sizes.insert(nrows, 0);
    }}
  }

  /**
   * Add a new element to the end of a specified row.
   *
   * - i: Row.
   * - x: Value.
   */
  function pushBack(i:Integer, x:Type) {
    auto j <- offsets[i] + sizes[i];
    cpp{{
    this_()->values.insert(j - 1, x);
    }}
    for k in (i + 1)..size() {
      offsets[k] <- offsets[k] + 1;
    }
    sizes[i] <- sizes[i] + 1;
  }

  /**
   * Iterate over the rows.
   *
   * Return: a fiber object that yields each row in forward order.
   */
  fiber walk() -> Type[_] {
    for i in 1..size() {
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
    for j in 1..sizes[i] {
      yield get(i, j);
    }
  }

  /**
   * First serial index of a row.
   *
   * - i: Row.
   */
  function from(i:Integer) -> Integer {
    assert 1 <= i && i <= size();
    assert offsets[i] != 0;  // not an empty row
    return offsets[i];
  }
  
  /**
   * Last serial index of a row.
   *
   * - i: Row.
   */
  function to(i:Integer) -> Integer {
    assert 1 <= i && i <= size();
    assert offsets[i] != 0;  // not an empty row
    return offsets[i] + sizes[i] - 1;
  }
  
  /**
   * Serial index of row and column.
   *
   * - i: Row.
   * - j: Column.
   */
  function serial(i:Integer, j:Integer) -> Integer {
    assert 1 <= i && i <= size();
    assert 1 <= j && j <= sizes[i];
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
