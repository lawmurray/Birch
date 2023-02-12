# Contributing

## Orientation

The main interface is defined in headers in the top-level `numbirch/` directory. It is divided by category:

* `array.hpp` copy-on-write array classes and functions,
* `memory.hpp` memory management functions,
* `numeric.hpp` linear algebra functions,
* `random.hpp` pseudorandom number generation,
* `reduce.hpp` reduction functions,
* `transform.hpp` element-wise transformation functions.

Subdirectories are then structured as follows:

* `array` backend-independent copy-on-write array implementation,
* `common` backend-independent common functionality,
* `instantiate` instantiation of all function templates,
* plus a subdirectory for each backend.

## Conventions

While NumBirch provides a template interface, its header files reveal only function template declarations and not function template definitions. Explicit instantiations are enumerated for all valid combinations of types for inclusion in the built library. This reduces build times for client code, as well as isolating the toolchains required for building backends such as CUDA and SYCL.

While many template functions are both declared and defined in `.hpp` header files, some are declared in a `.hpp` file but defined for each backend in separate `.inl` inline files. By convention, a `.inl` file is intended to be included by a `.cpp` source file or other `.inl` inline file, but never a `.hpp` header file. Notably they are included in the `.cpp` source files contained under `numbirch/instantiate`, with explicit template instantiations for all available types.

So, roughly speaking, we have:

* `.hpp` files declare the function templates,
* `.inl` files define the function templates,
* `.cpp` files instantiate the function templates.

There are functions that are not templates; these are in `.hpp` and `.cpp` files following the usual declare/define conventions.
