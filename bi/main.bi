hpp{{
#include "libbirch/libbirch.hpp"
#include "libubjpp/libubjpp.hpp"
}}

cpp{{
#ifdef __EMSCRIPTEN__
int main(int argc, char** argv) {
  typedef void prog_t(int argc, char** argv);
  
  std::string prog;
  if (argc <= 1) {
    std::cerr << "No program given" << std::endl;
  } else {
    std::string prog = argv[1];
    prog += '_';
    void* handle = dlopen(NULL, RTLD_NOW);
    void* addr = dlsym(handle, prog.c_str());
    if (addr) {
      prog_t* fcn = reinterpret_cast<prog_t*>(addr);
      fcn(argc - 1, argv + 1);
    } else {
      std::cerr << "Program " << argv[1] << " not found: " << dlerror() << std::endl;
    }
  }
  return 0;
}
#endif
}}
