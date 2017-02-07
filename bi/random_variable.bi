/**
 * Demonstrates the decay of a random variable, from holding a model to
 * being instantiated as a variate.
 */
program random_variable() {
  x:(Variate ~ Model);
  m:Model;
  
  x ~ m;
  f(x);
  decay(x);
  f(x);
}

model Variate {
  //
}

model Model {
  //
}

function f(x:Variate) {
  print("f(x:Variate)\n");
}

function f(x:Model) {
  print("f(x:Model)\n");
}

function x:Variate ~> m:Model {
  //
}

function ~m:Model -> x:Variate {
  //
}

/**
 * Forces a random variable to decay to a variate.
 */
function decay(x:Variate) {
  //
}
