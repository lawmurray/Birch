#include <iostream>
#include <dlfcn.h>

int main(int argc, char** argv) {
  if (argc > 1) {
    const char* prog = argc > 1 ? argv[1] : nullptr;
    void* addr = dlsym(RTLD_DEFAULT, prog);
    char* msg = dlerror();
    if (msg != nullptr) {
      std::cerr << "no program " << prog << std::endl;
      return 1;
    } else {
      typedef int prog_t(int argc, char** argv);
      prog_t* fcn = reinterpret_cast<prog_t*>(addr);
      return fcn(argc - 1, argv + 1);
    }
    return 0;
  } else {
    std::cerr << "no program specified" << std::endl;
  }
}
