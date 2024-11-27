#pragma once

#include <anns.hpp>
#include <vector>
#include <stdlib.h>
#include <omp.h>
#include <unordered_set>

namespace anns
{

  namespace utils
  {

    /// @brief get the recall of the results
    /// @param knn
    /// @param gt
    /// @param dgt the dimension of the gt vector (cause the `gt` is save in ivec(vector) file?)
    /// @return
    double recall(size_t k, size_t dgt, const knn_t &gt, const knn_t &knn)
    {
      assert(k <= dgt && "k must be less than or equal to dgt");
      const size_t nq = knn.size() / k;
      size_t ok = 0;
#pragma omp parallel for reduction(+ : ok)
      for (size_t q = 0; q < nq; q++)
      {
        std::unordered_set<int> st;
        for (size_t i = 0; i < k; i++)
        {
          st.insert(gt[q * dgt + i]);
        }
        for (size_t i = 0; i < k; i++)
        {
          if (st.count(knn[q * k + i]))
          {
            ok++;
          }
        }
      }
      return double(ok) / knn.size();
    }

    struct GroundTruth: public DataSetWrapper<int>
    {
      using DataSetWrapper<int>::base_;
      using DataSetWrapper<int>::dim_;
      using DataSetWrapper<int>::load;

      GroundTruth(const std::string &fname): DataSetWrapper<int>(fname) {}

      inline double recall(size_t k, const knn_t &knn) const noexcept
      {
        return utils::recall(k, dim_, base_, knn);
      }
      
    };

  }
}