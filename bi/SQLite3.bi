hpp{{
#include <sqlite3.h>
}}

/**
 * Wrapper for [sqlite3](https://www.sqlite.org/c3ref/sqlite3.html) struct.
 */
class SQLite3 {
  hpp{{
  sqlite3* db = nullptr;
  }}
  
  /**
   *
   */
  function open(filename:String) {
    cpp{{
    auto res = sqlite3_open(filename_.c_str(), &db);
    bi_assert_msg(res == SQLITE_OK, "sqlite3_open failed");
    }}
  }
  
  /**
   *
   */
  function close() {
    cpp{{
    auto res = sqlite3_close(db);
    bi_assert_msg(res == SQLITE_OK, "sqlite3_close failed");
    }}
  }
  
  /**
   * 
   */
  function prepare(sql:String) -> SQLite3Statement {
    stmt:SQLite3Statement;
    cpp{{
    auto res = sqlite3_prepare_v2(db, sql_.c_str(), sql_.length(), &stmt_->stmt, nullptr);
    bi_assert_msg(res == SQLITE_OK, "sqlite3_prepare_v2 failed");
    }}
    return stmt;
  }
}
