#pragma once
#include <cstdio>
#include <cstdlib>

// Invariant checks, compiled in only with the `checked` preset (CATAN_CHECKED=1).
// Failing a check prints the location and aborts.
#if CATAN_CHECKED
#define CATAN_CHECK(cond)                                                                 \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CATAN_CHECK failed: %s\n", __FILE__, __LINE__, \
                         #cond);                                                          \
            std::abort();                                                                 \
        }                                                                                 \
    } while (0)
#else
#define CATAN_CHECK(cond) ((void)0)
#endif
