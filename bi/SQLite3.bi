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
    auto res = sqlite3_open(filename_.c_str(), &self->db);
    bi_error_msg(res == SQLITE_OK, "sqlite3_open failed");
    }}
  }
  
  /**
   *
   */
  function close() {
    cpp{{
    auto res = sqlite3_close(self->db);
    bi_error_msg(res == SQLITE_OK, "sqlite3_close failed");
    }}
  }
  
  /**
   * 
   */
  function prepare(sql:String) -> SQLite3Statement {
    stmt:SQLite3Statement;
    cpp{{
    auto res = sqlite3_prepare_v2(self->db, sql_.c_str(), sql_.length(), &stmt_->stmt, nullptr);
    bi_error_msg(res == SQLITE_OK, "sqlite3_prepare_v2 failed");
    }}
    return stmt;
  }
  
  /**
   *
   */
  function exec(sql:String) {
    cpp{{
    auto res = sqlite3_exec(self->db, sql_.c_str(), nullptr, nullptr, nullptr);
    bi_error_msg(res == SQLITE_OK, "sqlite3_exec failed");
    }}
  }
}
