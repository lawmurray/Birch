/**
 * In-memory reader.
 */
class MemoryReader < Reader {
  hpp{{
  libubjpp::value* group = nullptr;
  }}  

  function getObject() -> Reader? {
    return getObject([]);
  }

  fiber getArray() -> Reader {
    /* fibers do not preserve variables declared in raw C++ code between
     * yields, and the switch and jump mechanism for resumption cannot jump
     * over variable initialization; so the value variable below is
     * recreated temporarily when needed, and put out of scope immediately
     * after */
    length:Integer;
    result:MemoryReader;
    cpp{{
    {
      auto value = this->self()->group->get<libubjpp::array_type>();
      length_ = value ? value.get().size() : 0;
    }
    }}
    for (i:Integer in 1..length) {
      cpp{{
      {
        auto value = this->self()->group->get<libubjpp::array_type>();
        assert(value);
        result_->group = &value.get()[i_ - 1];
      }
      }}
      yield result;
    }
  }

  function getLength() -> Integer? {
    return getLength([]);
  }

  function getBoolean() -> Boolean? {
    return getBoolean([]);
  }
  
  function getInteger() -> Integer? {
    return getInteger([]);
  }
  
  function getReal() -> Real? {
    return getReal([]);
  }
  
  function getString() -> String? {
    return getString([]);
  }

  function getBooleanVector() -> Boolean[_]? {
    return getBooleanVector([]);
  }
  
  function getIntegerVector() -> Integer[_]? {
    return getIntegerVector([]);
  }
  
  function getRealVector() -> Real[_]? {
    return getRealVector([]);
  }

  function getBooleanMatrix() -> Boolean[_,_]? {
    return getBooleanMatrix([]);
  }
  
  function getIntegerMatrix() -> Integer[_,_]? {
    return getIntegerMatrix([]);
  }
  
  function getRealMatrix() -> Real[_,_]? {
    return getRealMatrix([]);
  }

  function getObject(name:String) -> Reader? {
    return getObject([name]);
  }

  fiber getArray(name:String) -> Reader {
    length:Integer;
    result:MemoryReader;
    cpp{{
    {
      auto value = this->self()->group->get<libubjpp::array_type>(name_);
      length_ = value ? value.get().size() : 0;
    }
    }}
    for (i:Integer in 1..length) {
      cpp{{
      {
        auto value = this->self()->group->get<libubjpp::array_type>(name_);
        assert(value);
        result_->group = &value.get()[i_ - 1];
      }
      }}
      yield result;
    }
  }

  function getLength(name:String) -> Integer? {
    return getLength([name]);
  }

  function getBoolean(name:String) -> Boolean? {
    return getBoolean([name]);
  }
  
  function getInteger(name:String) -> Integer? {
    return getInteger([name]);
  }
  
  function getReal(name:String) -> Real? {
    return getReal([name]);
  }
  
  function getString(name:String) -> String? {
    return getString([name]);
  }

  function getBooleanVector(name:String) -> Boolean[_]? {
    return getBooleanVector([name]);
  }

  function getIntegerVector(name:String) -> Integer[_]? {
    return getIntegerVector([name]);
  }

  function getRealVector(name:String) -> Real[_]? {
    return getRealVector([name]);
  }

  function getBooleanMatrix(name:String) -> Boolean[_,_]? {
    return getBooleanMatrix([name]);
  }

  function getIntegerMatrix(name:String) -> Integer[_,_]? {
    return getIntegerMatrix([name]);
  }

  function getRealMatrix(name:String) -> Real[_,_]? {
    return getRealMatrix([name]);
  }

  function getObject(path:[String]) -> Reader? {
    exists:Boolean;
    cpp{{
    auto value = group->get(path_);
    exists_ = static_cast<bool>(value);
    }}
    if (exists) {
      result:MemoryReader;
      cpp{{
      result_->group = &value.get();
      }}
      return result;
    } else {
      return nil;
    }
  }

  fiber getArray(path:[String]) -> Reader {
    length:Integer;
    result:MemoryReader;
    cpp{{
    {
      auto value = this->self()->group->get<libubjpp::array_type>(path_);
      length_ = value ? value.get().size() : 0;
    }
    }}
    for (i:Integer in 1..length) {
      cpp{{
      {
        auto value = this->self()->group->get<libubjpp::array_type>(path_);
        assert(value);
        result_->group = &value.get()[i_ - 1];
      }
      }}
      yield result;
    }
  }

  function getLength(path:[String]) -> Integer? {
    cpp{{
    auto array = group->get<libubjpp::array_type>(path_);
    if (array) {
      return array.get().size();
    } else {
      return bi::nil;
    }
    }}
  }

  function getBoolean(path:[String]) -> Boolean? {
    cpp{{
    return group->get<libubjpp::bool_type>(path_);
    }}
  }
  
  function getInteger(path:[String]) -> Integer? {
    cpp{{
    return group->get<libubjpp::int64_type>(path_);
    }}
  }
  
  function getReal(path:[String]) -> Real? {
    value1:Real?;
    value2:Integer?;
    cpp{{
    value1_ = group->get<libubjpp::double_type>(path_);
    }}
    if (!value1?) {
      cpp{{
      value2_ = group->get<libubjpp::int64_type>(path_);
      }}
      value1 <- value2;
    }
    return value1;
  }

  function getString(path:[String]) -> String? {
    cpp{{
    return group->get<libubjpp::string_type>(path_);
    }}
  }

  function getBooleanVector(path:[String]) -> Boolean[_]? {
    length:Integer? <- getLength(path);
    if (length?) {
      cpp{{
      auto array = group->get<libubjpp::array_type>(path_).get();
      }}
      result:Boolean[length!];
      value:Boolean?;
      for (i:Integer in 1..length!) {
        cpp{{
        value_ = array[i_ - 1].get<libubjpp::bool_type>();
        }}
        if (value?) {
          result[i] <- value!;
        } else {
          return nil;
        }
      }
      return result;
    } else {
      return nil;
    }
  }

  function getIntegerVector(path:[String]) -> Integer[_]? {
    length:Integer? <- getLength(path);
    if (length?) {
      cpp{{
      auto array = group->get<libubjpp::array_type>(path_).get();
      }}
      result:Integer[length!];
      value:Integer?;
      for (i:Integer in 1..length!) {
        cpp{{
        value_ = array[i_ - 1].get<libubjpp::int64_type>();
        }}
        if (value?) {
          result[i] <- value!;
        } else {
          return nil;
        }
      }
      return result;
    } else {
      return nil;
    }
  }

  function getRealVector(path:[String]) -> Real[_]? {
    length:Integer? <- getLength(path);
    if (length?) {
      cpp{{
      auto array = group->get<libubjpp::array_type>(path_).get();
      }}
      result:Real[length!];
      value1:Real?;
      value2:Integer?;
      for (i:Integer in 1..length!) {
        cpp{{
        value1_ = array[i_ - 1].get<libubjpp::double_type>();
        }}
        if (value1?) {
          result[i] <- value1!;
        } else {
          cpp{{
          value2_ = array[i_ - 1].get<libubjpp::int64_type>();
          }}
          if (value2?) {
            result[i] <- value2!;
          } else {
            return nil;
          }
        }
      }
      return result;
    } else {
      return nil;
    }
  }

  function getBooleanMatrix(path:[String]) -> Boolean[_,_]? {
    nrows:Integer? <- getLength(path);
    if (nrows?) {
      row:Reader! <- getArray(path);
      if (row?) {
        /* determine number of columns from first row */
        ncols:Integer? <- row!.getLength();
        X:Boolean[_,_];
        x:Boolean[_]?;
        if (ncols?) {
          X <- matrix(false, nrows!, ncols!);
          x <- row!.getBooleanVector();
          if (x?) {
            X[1,1..ncols!] <- x!;
          } else {
            return nil;
          }
        }
        
        /* reader in remaining rows, requiring that they have the same
         * number of columns as the first */
        i:Integer <- 1;
        while (row?) {
          ncols <- row!.getLength();
          if (ncols? && ncols! == columns(X)) {
            x <- row!.getBooleanVector();
            if (x?) {
              i <- i + 1;
              X[i,1..ncols!] <- x!;
            } else {
              return nil;
            }
          } else {
            return nil;
          }
        }
        assert i == nrows!;
        return X;
      }
    }
    return nil;
  }

  function getIntegerMatrix(path:[String]) -> Integer[_,_]? {
    nrows:Integer? <- getLength(path);
    if (nrows?) {
      row:Reader! <- getArray(path);
      if (row?) {
        /* determine number of columns from first row */
        ncols:Integer? <- row!.getLength();
        X:Integer[_,_];
        x:Integer[_]?;
        if (ncols?) {
          X <- matrix(0, nrows!, ncols!);
          x <- row!.getIntegerVector();
          if (x?) {
            X[1,1..ncols!] <- x!;
          } else {
            return nil;
          }
        }
        
        /* reader in remaining rows, requiring that they have the same
         * number of columns as the first */
        i:Integer <- 1;
        while (row?) {
          ncols <- row!.getLength();
          if (ncols? && ncols! == columns(X)) {
            x <- row!.getIntegerVector();
            if (x?) {
              i <- i + 1;
              X[i,1..ncols!] <- x!;
            } else {
              return nil;
            }
          } else {
            return nil;
          }
        }
        assert i == nrows!;
        return X;
      }
    }
    return nil;
  }

  function getRealMatrix(path:[String]) -> Real[_,_]? {
    nrows:Integer? <- getLength(path);
    if (nrows?) {
      row:Reader! <- getArray(path);
      if (row?) {
        /* determine number of columns from first row */
        ncols:Integer? <- row!.getLength();
        X:Real[_,_];
        x:Real[_]?;
        if (ncols?) {
          X <- matrix(0.0, nrows!, ncols!);
          x <- row!.getRealVector();
          if (x?) {
            X[1,1..ncols!] <- x!;
          } else {
            return nil;
          }
        }
        
        /* reader in remaining rows, requiring that they have the same
         * number of columns as the first */
        i:Integer <- 1;
        while (row?) {
          ncols <- row!.getLength();
          if (ncols? && ncols! == columns(X)) {
            x <- row!.getRealVector();
            if (x?) {
              i <- i + 1;
              X[i,1..ncols!] <- x!;
            } else {
              return nil;
            }
          } else {
            return nil;
          }
        }
        assert i == nrows!;
        return X;
      }
    }
    return nil;
  }
}
