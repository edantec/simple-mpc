#include <Eigen/Core>

namespace simple_mpc
{
  namespace math
  {

    /// @note Backport from <pinocchio/math/matrix.hpp> in topic/simulation branch.
    template<typename Matrix>
    void make_symmetric(const Eigen::MatrixBase<Matrix> & mat, const int mode = Eigen::Upper)
    {
      if (mode == Eigen::Upper)
      {
        mat.const_cast_derived().template triangularView<Eigen::StrictlyLower>() =
          mat.transpose().template triangularView<Eigen::StrictlyLower>();
      }
      else if (mode == Eigen::Lower)
      {
        mat.const_cast_derived().template triangularView<Eigen::StrictlyUpper>() =
          mat.transpose().template triangularView<Eigen::StrictlyUpper>();
      }
    }

  } // namespace math
} // namespace simple_mpc
