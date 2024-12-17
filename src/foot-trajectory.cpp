///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "simple-mpc/foot-trajectory.hpp"
#include <ndcurves/bezier_curve.h>
#include <ndcurves/exact_cubic.h>
#include <ndcurves/se3_curve.h>
#include <ndcurves/so3_linear.h>
#include <pinocchio/spatial/se3.hpp>

namespace simple_mpc
{

  FootTrajectory::FootTrajectory(
    const std::map<std::string, point3_t> & initial_poses, double swing_apex, int T_fly, int T_contact, size_t T)
  {
    for (auto it = initial_poses.begin(); it != initial_poses.end(); it++)
    {
      references_.insert({it->first, std::vector<point3_t>(T, Eigen::Vector3d::Zero())});
    }
    initial_poses_ = initial_poses;
    final_poses_ = initial_poses;
    swing_apex_ = swing_apex;
    T_fly_ = T_fly;
    T_contact_ = T_contact;
    T_ = T;

    for (auto pose : initial_poses_)
    {
      piecewise_curve swing_trajectory = defineTranslationBezier(pose.second, pose.second);
      swing_trajectories_.insert({pose.first, swing_trajectory});
    }
  }

  piecewise_curve FootTrajectory::defineTranslationBezier(const point3_t & trans_init, const point3_t & trans_final)
  {
    std::vector<Eigen::Vector3d> points;
    for (long i = 0; i < 4; i++)
    { // init position. init vel,acc and jerk == 0
      points.push_back(trans_init);
    }
    // compute mid point (average and offset along z)
    Eigen::Vector3d midpoint = trans_init * 3 / 4 + trans_final * 1 / 4;
    midpoint[2] += swing_apex_;
    points.push_back(midpoint);
    for (long i = 5; i < 9; i++)
    { // final position. final vel,acc and jerk == 0
      points.push_back(trans_final);
    }
    curve_translation bezier_curve(curve_translation(points.begin(), points.end(), 0., 1.));

    piecewise_curve se3curve;
    se3curve.add_curve(bezier_curve);

    return se3curve;
  }

  std::vector<point3_t> FootTrajectory::createTrajectory(
    int time_to_land, point3_t & initial_trans, point3_t & final_trans, piecewise_curve trajectory_swing)
  {
    std::vector<point3_t> trajectory;
    for (int t = time_to_land; t > time_to_land - (int)T_; t--)
    {
      if (t < 0)
        trajectory.push_back(final_trans);
      else if (t > T_fly_)
        trajectory.push_back(initial_trans);
      else
      {
        point3_t trans = trajectory_swing(float(T_fly_ - t) / float(T_fly_));
        trajectory.push_back(trans);
      }
    }

    return trajectory;
  }

  void FootTrajectory::updateTrajectory(
    bool update, int landing_time, const point3_t & ee_trans, const point3_t & final_trans, const std::string & ee_name)
  {
    if (update)
    {
      initial_poses_.at(ee_name) = ee_trans;
      final_poses_.at(ee_name) = final_trans;
      swing_trajectories_.at(ee_name) = defineTranslationBezier(initial_poses_.at(ee_name), final_poses_.at(ee_name));
    }

    references_.at(ee_name) = createTrajectory(
      landing_time, initial_poses_.at(ee_name), final_poses_.at(ee_name), swing_trajectories_.at(ee_name));
  }

} // namespace simple_mpc
