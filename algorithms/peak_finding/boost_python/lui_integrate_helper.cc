#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <dials/algorithms/peak_finding/lui_integrate_helper.h>

namespace dials { namespace algorithms { namespace boost_python {

  using namespace boost::python;

  void export_lui_integrate_helper() {
    def("model_2d", &model_2d, (arg("nrow")=100, arg("ncol")=100,
    arg("a") = 10, arg("b") = 20, arg("delta_ang") = 1,
    arg("imax") = 50 ,   arg("asp") = 0.5 ) );

    def("measure_2d_angl", &measure_2d_angl, ( arg("data2d"),
        arg("xpos")=1, arg("ypos")=20 ) );

  }

}}} // namespace = dials::algorithms::boost_python
