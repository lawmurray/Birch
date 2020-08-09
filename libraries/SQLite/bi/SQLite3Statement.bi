/**
 * Wrapper for [sqlite3_stmt](https://www.sqlite.org/c3ref/stmt.html) struct.
 */
class SQLite3Statement {
  hpp{{
  sqlite3_stmt* stmt = nullptr;
  }}

  /**
   * Bind argument to a parameter.
   *
   * - i: Parameter index, 1-based.
   * - x: Argument.
   */ 
  function bind(i:Integer, x:Integer) {
    cpp{{
    auto res = sqlite3_bind_int64(this_()->stmt, i, x);
    libbirch_error_msg_(res == SQLITE_OK, "sqlite3_bind_int64 failed");
    }}
  }

  /**
   * Bind argument to a parameter.
   *
   * - i: Parameter index, 1-based.
   * - x: Argument.
   */ 
  function bind(i:Integer, x:Real) {
    cpp{{
    auto res = sqlite3_bind_double(this_()->stmt, i, x);
    libbirch_error_msg_(res == SQLITE_OK, "sqlite3_bind_double failed");
    }}
  }

  /**
   * Bind argument to a parameter.
   *
   * - i: Parameter index, 1-based.
   * - x: Argument.
   */ 
  function bind(i:Integer, x:String) {
    cpp{{
    auto res = sqlite3_bind_text(this_()->stmt, i, x.c_str(), x.length(), SQLITE_TRANSIENT);
    libbirch_error_msg_(res == SQLITE_OK, "sqlite3_bind_text failed");
    }}
  }

  /**
   * Bind null argument to a parameter.
   *
   * - i: Parameter index, 1-based.
   */ 
  function bindNull(i:Integer) {
    cpp{{
    auto res = sqlite3_bind_null(this_()->stmt, i);
    libbirch_error_msg_(res == SQLITE_OK, "sqlite3_bind_null failed");
    }}
  }

  /**
   * Execute the statement one step. For a query, this returns true if there
   * is a new row to read (using the `column*` functions) and false if there
   * are no more rows to read. For an update, this returns false on success
   * (as there are no results to read). Anything else throws an error.
   *
   * Intenally, this is a simplified version of
   * [sqlite3_step](https://www.sqlite.org/c3ref/step.html). If
   * `sqlite3_step` returns:
   *
   *   * `SQL_ROW` it returns true,
   *   * `SQL_DONE` it returns false, and
   *   * anything else, throws and error.
   */
  function step() -> Boolean {
    cpp{{
    auto res = sqlite3_step(this_()->stmt);
    libbirch_error_msg_(res == SQLITE_ROW || res == SQLITE_DONE, "sqlite3_step failed");
    return (res == SQLITE_ROW);
    }}
  }
  
  /**
   * Number of columns in the result.
   */
  function columnCount() -> Integer {
    cpp{{
    return sqlite3_column_count(this_()->stmt);
    }}
  }
  
  /**
   * Get column value as an integer.
   *
   * - i: Column index, 1-based.
   *
   * Return: Optional with a value if the column value is a non-null integer.
   */ 
  function columnInteger(i:Integer) -> Integer? {
    cpp{{
    if (sqlite3_column_type(this_()->stmt, i - 1) == SQLITE_INTEGER) {
      return (Integer)sqlite3_column_int64(this_()->stmt, i - 1);
    }
    }}
    return nil;
  }

  /**
   * Get column value as a real.
   *
   * - i: Column index, 1-based.
   *
   * Return: Optional with a value if the column value is a non-null real.
   */ 
  function columnReal(i:Integer) -> Real? {
    cpp{{
    if (sqlite3_column_type(this_()->stmt, i - 1) == SQLITE_FLOAT) {
      return sqlite3_column_double(this_()->stmt, i - 1);
    }
    }}
    return Real?(columnInteger(i));
  }

  /**
   * Get column value as a string.
   *
   * - i: Column index, 1-based.
   *
   * Return: Optional with a value if the column value is a non-null string.
   */ 
  function columnString(i:Integer) -> String? {
    cpp{{
    if (sqlite3_column_type(this_()->stmt, i - 1) == SQLITE_TEXT) {
      return std::string(reinterpret_cast<const char*>(sqlite3_column_text(this_()->stmt, i - 1)));
    }
    }}
    return nil;
  }

  /**
   * Is the column value `null`?
   *
   * - i: Column index, 1-based.
   */
  function columnNull(i:Integer) -> Boolean {
    cpp{{
    return sqlite3_column_type(this_()->stmt, i - 1) == SQLITE_NULL;
    }}
  }

  /**
   * Reset the statement for reuse.
   */
  function reset() {
    cpp{{
    auto res = sqlite3_reset(this_()->stmt);
    libbirch_error_msg_(res == SQLITE_OK, "sqlite3_reset failed");
    }}
  }
   
  /**
   * Finalize the statement.
   */
  function finalize() {
    cpp{{
    auto res = sqlite3_finalize(this_()->stmt);
    libbirch_error_msg_(res == SQLITE_OK, "sqlite3_finalize failed");
    }}
  }
}
