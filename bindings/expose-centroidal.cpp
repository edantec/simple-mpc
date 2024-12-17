#include "simple-mpc/centroidal-dynamics.hpp"
#include "simple-mpc/python.hpp"

#include <eigenpy/std-map.hpp>

namespace simple_mpc::python
{

  auto * createCentroidal(const bp::dict & settings, const RobotModelHandler & model_handler)
  {
    CentroidalSettings conf;
    conf.timestep = bp::extract<double>(settings["timestep"]);
    conf.w_com = bp::extract<Eigen::Matrix3d>(settings["w_com"]);
    conf.w_u = bp::extract<Eigen::MatrixXd>(settings["w_u"]);
    conf.w_linear_mom = bp::extract<Eigen::Matrix3d>(settings["w_linear_mom"]);
    conf.w_angular_mom = bp::extract<Eigen::Matrix3d>(settings["w_angular_mom"]);
    conf.w_linear_acc = bp::extract<Eigen::Matrix3d>(settings["w_linear_acc"]);
    conf.w_angular_acc = bp::extract<Eigen::Matrix3d>(settings["w_angular_acc"]);

    conf.gravity = bp::extract<Eigen::Vector3d>(settings["gravity"]);
    conf.force_size = bp::extract<int>(settings["force_size"]);

    conf.mu = bp::extract<double>(settings["mu"]);
    conf.Lfoot = bp::extract<double>(settings["Lfoot"]);
    conf.Wfoot = bp::extract<double>(settings["Wfoot"]);

    return new CentroidalOCP(conf, model_handler);
  }

  StageModel createCentStage(
    CentroidalOCP & self,
    const bp::dict & phase_dict,
    const bp::dict & pose_dict,
    const bp::dict & force_dict,
    const bp::dict & land_dict)
  {
    bp::list phase_keys = bp::list(phase_dict.keys());
    bp::list pose_keys = bp::list(pose_dict.keys());
    bp::list force_keys = bp::list(force_dict.keys());
    bp::list land_keys = bp::list(land_dict.keys());
    std::map<std::string, bool> phase_contact;
    std::map<std::string, pinocchio::SE3> pose_contact;
    std::map<std::string, Eigen::VectorXd> force_contact;
    std::map<std::string, bool> land_constraint;
    for (int i = 0; i < len(phase_keys); ++i)
    {
      bp::extract<std::string> extractor(phase_keys[i]);
      if (extractor.check())
      {
        std::string key = extractor();
        bool ff = bp::extract<bool>(phase_dict[key]);
        phase_contact.insert({key, ff});
      }
    }
    for (int i = 0; i < len(pose_keys); ++i)
    {
      bp::extract<std::string> extractor(pose_keys[i]);
      if (extractor.check())
      {
        std::string key = extractor();
        pinocchio::SE3 ff = bp::extract<pinocchio::SE3>(pose_dict[key]);
        pose_contact.insert({key, ff});
      }
    }
    for (int i = 0; i < len(force_keys); ++i)
    {
      bp::extract<std::string> extractor(force_keys[i]);
      if (extractor.check())
      {
        std::string key = extractor();
        Eigen::VectorXd ff = bp::extract<Eigen::VectorXd>(force_dict[key]);
        force_contact.insert({key, ff});
      }
    }
    for (int i = 0; i < len(land_keys); ++i)
    {
      bp::extract<std::string> extractor(land_keys[i]);
      if (extractor.check())
      {
        std::string key = extractor();
        bool ff = bp::extract<bool>(land_dict[key]);
        land_constraint.insert({key, ff});
      }
    }

    return self.createStage(phase_contact, pose_contact, force_contact, land_constraint);
  }

  bp::dict getSettingsCent(CentroidalOCP & self)
  {
    CentroidalSettings conf = self.getSettings();
    bp::dict settings;
    settings["timestep"] = conf.timestep;
    settings["w_com"] = conf.w_com;
    settings["w_u"] = conf.w_u;
    settings["w_linear_mom"] = conf.w_linear_mom;
    settings["w_angular_mom"] = conf.w_angular_mom;
    settings["w_linear_acc"] = conf.w_linear_acc;
    settings["w_angular_acc"] = conf.w_angular_acc;
    settings["gravity"] = conf.gravity;
    settings["mu"] = conf.mu;
    settings["Lfoot"] = conf.Lfoot;
    settings["Wfoot"] = conf.Wfoot;
    settings["force_size"] = conf.force_size;

    return settings;
  }

  void exposeCentroidalOcp()
  {
    bp::register_ptr_to_python<std::shared_ptr<CentroidalOCP>>();

    bp::class_<CentroidalOCP, bp::bases<OCPHandler>, boost::noncopyable>("CentroidalOCP", bp::no_init)
      .def(
        "__init__",
        bp::make_constructor(&createCentroidal, bp::default_call_policies(), ("settings"_a, "model_handler")))
      .def("getSettings", &getSettingsCent)
      .def("createStage", &createCentStage);
  }

} // namespace simple_mpc::python
