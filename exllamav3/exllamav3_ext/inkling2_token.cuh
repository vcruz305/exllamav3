#pragma once

#include <pybind11/pybind11.h>

// Inkling2Loop whole-token decode (see inkling2_token.cu): binds the InkToken class
void ink_token_bind(pybind11::module_& m);
