/**
 * Buffer in memory.
 */
class MemoryBuffer < Buffer {
  value:Value;

  /**
   * Load from a file.
   *
   * - path: Path of the file.
   */
  function load(path:String) {
    parser:JSONParser;
    value <- parser.parse(path);
    if !root {
      stderr.print("warning: could not load from \'" + path + "'\n");
    }
  }

  /**
   * Save to a file.
   *
   * - path: Path of the file.
   */
  function save(path:String) {
    mkdir(path);
    generator:JSONGenerator;
    auto success <- generator.generate(path, value);
    if !success {
      stderr.print("warning: could not save to \'" + path + "'\n");
    }
  }

  function getChild(name:String) -> Buffer? {
    return value.getChild();
  }

  function getObject() -> Buffer? {
    return value.getObject();
  }

  function getLength() -> Integer? {
    return value.getLength();
  }

  function getArray() -> Buffer? {
    return value.getArray();
  }

  function getBoolean() -> Boolean? {
    return value.getBoolean();
  }
  
  function getInteger() -> Integer? {
    return value.getInteger();
  }
  
  function getReal() -> Real? {
    return value.getReal();
  }

  function getString() -> String? {
    return value.getString();
  }

  function getObject(value:Object) -> Object? {
    value.read(this);
    return value;
  }

  function getBooleanVector() -> Boolean[_]? {
    return value.getBooleanVector();
  }

  function getIntegerVector() -> Integer[_]? {
    return value.getIntegerVector();
  }

  function getRealVector() -> Real[_]? {
    return value.getRealVector();
  }

  function getBooleanMatrix() -> Boolean[_,_]? {
    return value.getBooleanMatrix();
  }

  function getIntegerMatrix() -> Integer[_,_]? {
    return value.getIntegerMatrix();
  }

  function getRealMatrix() -> Real[_,_]? {
    return value.getRealMatrix();
  }

  function setChild(name:String) -> Buffer {
    return value.setChild();
  }

  function setObject() {
    v:ObjectValue;
    this.value <- v;
  }

  function setArray() {
    v:ArrayValue;
    this.value <- v;
  }
  
  function setNil() {
    v:NilValue;
    this.value <- v;
  }
  
  function setBoolean(value:Boolean?) {
    if value? {
      v:BooleanValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }
  
  function setInteger(value:Integer?) {
    if value? {
      v:IntegerValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }
  
  function setReal(value:Real?) {
    if value? {
      v:RealValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }

  function setString(value:String?) {
    if value? {
      v:StringValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }

  function setObject(value:Object?) {
    if value? {
      v:ObjectValue(value!);
      value!.write(this);
    } else {
      setNil();
    }
  }

  function setBooleanVector(value:Boolean[_]?) {
    if value? {
      v:BooleanVectorValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }
  
  function setIntegerVector(value:Integer[_]?) {
    if value? {
      v:IntegerVectorValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }
  
  function setRealVector(value:Real[_]?) {
    if value? {
      v:RealVectorValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }

  function setBooleanMatrix(value:Boolean[_,_]?) {
    if value? {
      v:BooleanMatrixValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }
  
  function setIntegerMatrix(value:Integer[_,_]?) {
    if value? {
      v:IntegerMatrixValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }
  
  function setRealMatrix(value:Real[_,_]?) {
    if value? {
      v:RealMatrixValue(value!);
      this.value <- v;
    } else {
      setNil();
    }
  }
}
