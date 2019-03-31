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
    auto res = sqlite3_open(filename.c_str(), &self->db);
    libbirch_error_msg_(res == SQLITE_OK, "sqlite3_open failed");
    }}
  }
  
  /**
   *
   */
  function close() {
    cpp{{
    auto res = sqlite3_close(self->db);
    libbirch_error_msg_(res == SQLITE_OK, "sqlite3_close failed");
    }}
  }
  
  /**
   * 
   */
  function prepare(sql:String) -> SQLite3Statement {
    stmt:SQLite3Statement;
    cpp{{
    auto res = sqlite3_prepare_v2(self->db, sql.c_str(), sql.length(), &stmt->stmt, nullptr);
    libbirch_error_msg_(res == SQLITE_OK, "sqlite3_prepare_v2 failed");
    }}
    return stmt;
  }
  
  /**
   *
   */
  function exec(sql:String) {
    cpp{{
    auto res = sqlite3_exec(self->db, sql.c_str(), nullptr, nullptr, nullptr);
    libbirch_error_msg_(res == SQLITE_OK, "sqlite3_exec failed");
    }}
  }
}
