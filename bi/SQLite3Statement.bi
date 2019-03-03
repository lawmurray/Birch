/**
 * Wrapper for [sqlite3_stmt](https://www.sqlite.org/c3ref/stmt.html) struct.
 */
class SQLite3Statement {
  hpp{{
  sqlite3_stmt* stmt = nullptr;
  }}
  
  function finalize() {
    cpp{{
    auto res = sqlite3_finalize(stmt);
    bi_assert_msg(res == SQLITE_OK, "sqlite3_finalize failed");
    }}
  }
}
