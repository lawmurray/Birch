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
    assert(!this_()->db);
    auto res = sqlite3_open(filename.c_str(), &this_()->db);
    libbirch_error_msg_(res == SQLITE_OK, "sqlite3_open failed");
    }}
  }
  
  /**
   *
   */
  function close() {
    cpp{{
    auto res = sqlite3_close(this_()->db);
    libbirch_error_msg_(res == SQLITE_OK, "sqlite3_close failed");
    this_()->db = nullptr;
    }}
  }
  
  /**
   * 
   */
  function prepare(sql:String) -> SQLite3Statement {
    stmt:SQLite3Statement;
    cpp{{
    auto res = sqlite3_prepare_v2(this_()->db, sql.c_str(), sql.length(), &stmt->stmt, nullptr);
    libbirch_error_msg_(res == SQLITE_OK, "sqlite3_prepare_v2 failed");
    }}
    return stmt;
  }
  
  /**
   *
   */
  function exec(sql:String) {
    cpp{{
    auto res = sqlite3_exec(this_()->db, sql.c_str(), nullptr, nullptr, nullptr);
    libbirch_error_msg_(res == SQLITE_OK, "sqlite3_exec failed");
    }}
  }
}
