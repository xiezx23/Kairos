#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lut.h"

PYBIND11_MODULE(LUT_module, m) {
    pybind11::class_<LUT>(m, "LUT")
        .def(pybind11::init<>())
        .def("get", &LUT::get, "Get the best component for given value",
             pybind11::arg("m"))
        .def("record", &LUT::record, "Record a new segment and component",
             pybind11::arg("r"), pybind11::arg("comp_type"))
        .def("shrink_to_fit", &LUT::shrink_to_fit, 
             "Optimize memory usage after all records");
}