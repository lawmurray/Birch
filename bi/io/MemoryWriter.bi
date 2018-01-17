/**
 * In-memory writer.
 */
class MemoryWriter < Writer {
  hpp{{
  libubjpp::value* group;
  }}

  function put(name:String, value:Boolean) {
    cpp{{
    group->set(name_, value_);
    }}
  }
  
  function put(name:String, value:Integer) {
    cpp{{
    group->set(name_, value_);
    }}
  }
  
  function put(name:String, value:Real) {
    cpp{{
    group->set(name_, value_);
    }}
  }
  
  function put(path:[String], value:Boolean) {
    cpp{{
    group->set(path_, value_);
    }}
  }
  
  function put(path:[String], value:Integer) {
    cpp{{
    group->set(path_, value_);
    }}
  }
  
  function put(path:[String], value:Real) {
    cpp{{
    group->set(path_, value_);
    }}
  }
}
