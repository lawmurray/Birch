/**
 * Wrapper for [sqlite3_stmt](https://www.sqlite.org/c3ref/stmt.html) struct.
 */
class SQLite3Statement {
  hpp{{
  sqlite3_stmt* stmt = nullptr;
  }}
  
  function bind(i:Integer32, x:Integer32) {
    cpp{{
    auto res = sqlite3_bind_int(stmt, i_, x_);
    bi_assert_msg(res == SQLITE_OK, "sqlite3_bind_int failed");
    }}
  }

  function bind(i:Integer32, x:Integer) {
    cpp{{
    auto res = sqlite3_bind_int64(stmt, i_, x_);
    bi_assert_msg(res == SQLITE_OK, "sqlite3_bind_int64 failed");
    }}
  }

  function bind(i:Integer32, x:Real) {
    cpp{{
    auto res = sqlite3_bind_double(stmt, i_, x_);
    bi_assert_msg(res == SQLITE_OK, "sqlite3_bind_double failed");
    }}
  }

  function bind(i:Integer32, x:String) {
    cpp{{
    auto res = sqlite3_bind_text(stmt, i_, x_.c_str(), x_.length(), SQLITE_TRANSIENT);
    bi_assert_msg(res == SQLITE_OK, "sqlite3_bind_text failed");
    }}
  }

  function bindNull(i:Integer32) {
    cpp{{
    auto res = sqlite3_bind_null(stmt, i_);
    bi_assert_msg(res == SQLITE_OK, "sqlite3_bind_null failed");
    }}
  }
  
  function finalize() {
    cpp{{
    auto res = sqlite3_finalize(stmt);
    bi_assert_msg(res == SQLITE_OK, "sqlite3_finalize failed");
    }}
  }
}
