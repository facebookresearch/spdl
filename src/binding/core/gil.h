#pragma once

#include <nanobind/nanobind.h>

#ifdef SPDL_HOLD_GIL
#define RELEASE_GIL()
#else
#define RELEASE_GIL() nb::gil_scoped_release __g;
#endif
