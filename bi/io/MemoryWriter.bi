/**
 * In-memory writer.
 */
class MemoryWriter < Writer {
  hpp{{
  libubjpp::value* group;
  }}

  function set(name:String, value:Boolean) {
    cpp{{
    group->set(name_, value_);
    }}
  }
  
  function set(name:String, value:Integer) {
    cpp{{
    group->set(name_, value_);
    }}
  }
  
  function set(name:String, value:Real) {
    cpp{{
    group->set(name_, value_);
    }}
  }
  
  function set(path:[String], value:Boolean) {
    cpp{{
    group->set(path_, value_);
    }}
  }
  
  function set(path:[String], value:Integer) {
    cpp{{
    group->set(path_, value_);
    }}
  }
  
  function set(path:[String], value:Real) {
    cpp{{
    group->set(path_, value_);
    }}
  }
}
