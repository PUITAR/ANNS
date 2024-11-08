#pragma once

#include <anns.hpp>
#include <vector>
#include <stdlib.h>
#include <omp.h>
#include <unordered_set>

namespace utils {

/// @brief get the recall of the results
/// @param knn_results 
/// @param ground_truth 
/// @param dimension_gt the dimension of the ground_truth vector (cause the `ground_truth` is save in ivec(vector) file?)
/// @return 
double recall(size_t k, size_t dimension_gt, const std::vector<id_t> & ground_truth, const std::vector<std::vector<id_t>> & knn_results) {
  const size_t nq = knn_results.size();
  size_t ok = 0, neg = 0;
  #pragma omp parallel for reduction(+: ok, neg)
  for (size_t q = 0; q < nq; q++) {
    // const size_t actual_k = knn_results[q].size();
    std::unordered_set<id_t> st; // (ground_truth.begin() + q * dimension_gt, ground_truth.begin() + q * dimension_gt + k);
    for (size_t i = 0; i < k; i++)
    {
      id_t id = ground_truth[q * dimension_gt + i];
      if (id != MAGIC_ID)
      {
        st.insert(id);
      }
      else {
        neg ++;
      }
    }
    for (const auto & id : knn_results[q]) {
      if (st.count(id)) {
        ok ++;
      }
    }
  }
  return double(ok) / (nq * k - neg);
}

}