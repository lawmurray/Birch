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
}
