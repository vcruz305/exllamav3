// Python binding of ExlMoeRow (exl3_moe_row.cu): class ext.MoeRow. Registered from bindings.cpp via exl3_moe_row_bind.
#include <torch/extension.h>
#include "exl3_moe_row.cuh"

void exl3_moe_row_bind(pybind11::module_& m)
{
    namespace py = pybind11;
    py::class_<ExlMoeRow>(m, "MoeRow")
        .def(py::init<int, int, int, int, at::Tensor, at::Tensor, int, float, float, at::Tensor,
                      c10::optional<at::Tensor>, float>(),
             py::arg("H"), py::arg("I"), py::arg("E"), py::arg("top_k"), py::arg("ptrs"), py::arg("kt"),
             py::arg("cb"), py::arg("cap_g"), py::arg("cap_u"), py::arg("gate_t"), py::arg("bias_f32"),
             py::arg("route_scale"))
        .def("forward", &ExlMoeRow::forward, py::arg("y"), py::arg("out"))
        .def("run_experts", &ExlMoeRow::run_experts, py::arg("y"), py::arg("sel"), py::arg("w"), py::arg("out"))
        .def_readonly("logits", &ExlMoeRow::logits)
        .def_readonly("sel", &ExlMoeRow::sel)
        .def_readonly("w", &ExlMoeRow::w);
}
