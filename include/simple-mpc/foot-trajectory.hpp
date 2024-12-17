///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SIMPLE_MPC_FOOTTRAJ_HPP_
#define SIMPLE_MPC_FOOTTRAJ_HPP_

#include "simple-mpc/fwd.hpp"
#include <ndcurves/fwd.h>
#include <ndcurves/piecewise_curve.h>

namespace simple_mpc
{
  /**
   * @brief Foot trajectory generation
   */

  typedef Eigen::Vector3d point3_t;
  typedef ndcurves::bezier_curve<float, double, false, point3_t> curve_translation;
  typedef ndcurves::piecewise_curve<float, double, false, point3_t> piecewise_curve;

  class FootTrajectory
  {
    /**
     */
  protected:
    std::map<std::string, point3_t> initial_poses_;
    std::map<std::string, point3_t> final_poses_;
    std::map<std::string, std::vector<point3_t>> references_;
    double swing_apex_;
    std::map<std::string, piecewise_curve> swing_trajectories_;
    int T_fly_;
    int T_contact_;
    size_t T_;

  public:
    FootTrajectory() {};
    virtual ~FootTrajectory() {};
    FootTrajectory(
      const std::map<std::string, point3_t> & initial_poses, double swing_apex, int T_fly, int T_contact, size_t T);

    void updateApex(double swing_apex)
    {
      swing_apex_ = swing_apex;
    }
    piecewise_curve defineTranslationBezier(const point3_t & trans_init, const point3_t & trans_final);
    std::vector<point3_t> createTrajectory(
      int time_to_land, point3_t & initial_trans, point3_t & final_trans, piecewise_curve trajectory_swing);
    void updateTrajectory(
      bool update,
      int landing_time,
      const point3_t & ee_trans,
      const point3_t & final_trans,
      const std::string & ee_name);
    const std::vector<point3_t> & getReference(const std::string & ee_name)
    {
      return references_.at(ee_name);
    }
  };

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_HPP_
