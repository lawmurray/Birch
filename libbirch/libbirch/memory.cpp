/**
 * @file
 */
#include "libbirch/memory.hpp"

#include "libbirch/thread.hpp"
#include "libbirch/Atomic.hpp"
#include "libbirch/Any.hpp"
#include "libbirch/Marker.hpp"
#include "libbirch/Scanner.hpp"
#include "libbirch/Collector.hpp"
#include "libbirch/Spanner.hpp"

/**
 * Possible roots list for each thread.
 */
static thread_local std::vector<libbirch::Any*> possible_roots;

/**
 * Unreachable list for each thread.
 */
static thread_local std::vector<libbirch::Any*> unreachable;

/**
 * Biconnected flag for each thread.
 */
static thread_local bool biconnected_flag = false;

void libbirch::register_possible_root(Any* o) {
  possible_roots.push_back(o);
}

void libbirch::register_unreachable(Any* o) {
  unreachable.push_back(o);
}

void libbirch::collect() {
  /* concatenates the possible roots list of each thread into a single list,
   * then distributes the passes over this list between all threads; this
   * improves load balancing over each thread operating only on its original
   * list */

  auto nthreads = get_max_threads();
  std::vector<libbirch::Any*> all_possible_roots;   // concatenated list of possible roots
  std::vector<int> starts(nthreads); // start indices in concatenated list
  std::vector<int> sizes(nthreads);  // sizes in concatenated list

  #pragma omp parallel
  {
    auto tid = get_thread_num();

    /* objects can be added to the possible roots list during normal
     * execution, but not removed, although they may be flagged as no longer
     * being a possible root; remove such objects first */
    int size = 0;
    for (int i = 0; i < (int)possible_roots.size(); ++i) {
      auto o = possible_roots[i];
      if (o->isPossibleRoot_()) {
        possible_roots[size++] = o;
      } else if (o->numShared_() == 0) {
        o->deallocate_();  // deallocation was deferred until now
      } else {
        o->unbuffer_();  // not a root, mark as no longer in the buffer
      }
    }
    possible_roots.resize(size);
    sizes[tid] = size;
    #pragma omp barrier

    /* a single thread now sets up the concatenated list of possible roots */
    #pragma omp single
    {
      #ifdef __cpp_lib_parallel_algorithm
      std::exclusive_scan(sizes.begin(), sizes.end(), starts.begin(), 0);
      #else
      starts[0] = 0;
      for (int i = 1; i < nthreads; ++i) {
        starts[i] = starts[i - 1] + sizes[i - 1];
      }
      #endif
      all_possible_roots.resize(starts.back() + sizes.back());
    }
    #pragma omp barrier

    /* all threads copy into the concatenated list of possible roots */
    std::copy(possible_roots.begin(), possible_roots.end(),
        all_possible_roots.begin() + starts[tid]);
    possible_roots.clear();
    #pragma omp barrier

    /* mark pass */
    #pragma omp for schedule(static)
    for (int i = 0; i < (int)all_possible_roots.size(); ++i) {
      auto o = all_possible_roots[i];
      Marker visitor;
      visitor.visit(o);
    }
    #pragma omp barrier

    /* scan/reach pass */
    #pragma omp for schedule(static)
    for (int i = 0; i < (int)all_possible_roots.size(); ++i) {
      auto o = all_possible_roots[i];
      Scanner visitor;
      visitor.visit(o);
    }
    #pragma omp barrier

    /* collect pass */
    #pragma omp for schedule(static)
    for (int i = 0; i < (int)all_possible_roots.size(); ++i) {
      auto o = all_possible_roots[i];
      Collector visitor;
      visitor.visit(o);
    }
    #pragma omp barrier

    /* finally, destroy objects determined unreachable */
    for (int i = 0; i < (int)unreachable.size(); ++i) {
      auto o = unreachable[i];
      o->destroy_();
      o->deallocate_();
    }
    unreachable.clear();
  }
}

bool libbirch::biconnected_copy(const bool toggle) {
  if (toggle) {
    biconnected_flag = !biconnected_flag;
  }
  return biconnected_flag;
}
