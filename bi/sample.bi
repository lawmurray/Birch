import VBD;

program sample(T:Integer <- 10) {
  x:VBD(T);
  f:Real! <- x.simulate();
  while (f?) {
    //f!;
  }
  x.output();
}
