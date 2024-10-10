#pragma once

#include <graph/hcnng.hpp>
#include <interval/interval.hpp>

using namespace anns::graph;

namespace anns
{

  namespace interval
  {

    namespace graph
    {

      template <typename data_t, typename label_t, float (*distance)(const data_t *, const data_t *, size_t)>

      class PostFilterHCNNG : public HCNNG<data_t, distance>
      {
      public:
        std::vector<Interval<label_t>> ft_;

        PostFilterHCNNG(size_t dim, size_t num_random_clusters, size_t min_size_clusters, size_t max_mst_degree) noexcept : HCNNG<data_t, distance>(dim, num_random_clusters, min_size_clusters, max_mst_degree) {}

        PostFilterHCNNG(const)

            void set_intervals(const std::vector<Interval<label_t>> &ft) noexcept
        {
          assert(ft.size() == this->cur_element_count_);
          ft_ = ft;
        }

        void hard_search(
            const std::vector<std::vector<data_t>> &queries,
            size_t k,
            size_t ef,
            const std::vector<Interval<label_t>> &qft,
            std::vector<std::vector<id_t>> &knn,
            std::vector<std::vector<float>> &dists)
        {
          assert(queries.size() == qft.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(this->num_threads_)
          for (size_t i = 0; i < queries.size(); i++)
          {
            const data_t *qi = queries[i].data();
            size_t actual_k = k * 2;
            auto &knn_i = knn[i];
            auto &dists_i = dists[i];
            decltype(knn_i) knn_i_temp(knn_i.size());
            decltype(dists_i) dists_i_temp(dists_i.size());
            do
            {
              knn_i_temp.clear(), dists_i_temp.clear();
              auto res = this->search_base_layer(qi, actual_k, std::max(ef, actual_k));
              while (res.size() && knn_i_temp.size() < k)
              {
                auto [d, id] = res.top();
                if (ft_[id].cover(qft[i]))
                  knn_i_temp.emplace_back(tt.second), dists_i_temp.emplace_back(tt.first);
                res.pop();
              }
              actual_k *= 2;
            } while (knn_i_temp.size() < k);
            knn_i = std::move(knn_i_temp);
            dists_i = std::move(dists_i_temp);
          }
        }

        void soft_search(
            const std::vector<std::vector<data_t>> &queries,
            size_t k,
            size_t ef,
            const std::vector<Interval<label_t>> &qft,
            std::vector<std::vector<id_t>> &knn,
            std::vector<std::vector<float>> &dists)
        {
          assert(queries.size() == qft.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(this->num_threads_)
          for (size_t i = 0; i < queries.size(); i++)
          {
            const data_t *qi = queries[i].data();
            size_t actual_k = k * 2;
            auto &knn_i = knn[i];
            auto &dists_i = dists[i];
            decltype(knn_i) knn_i_temp(knn_i.size());
            decltype(dists_i) dists_i_temp(dists_i.size());
            do
            {
              knn_i_temp.clear(), dists_i_temp.clear();
              auto res = this->search_base_layer(qi, actual_k, std::max(ef, actual_k));
              while (res.size() && knn_i_temp.size() < k)
              {
                auto [d, id] = res.top();
                if (ft_[id].overlap(qft[i]))
                  knn_i_temp.emplace_back(tt.second), dists_i_temp.emplace_back(tt.first);
                res.pop();
              }
              actual_k *= 2;
            } while (knn_i_temp.size() < k);
            knn_i = std::move(knn_i_temp);
            dists_i = std::move(dists_i_temp);
          }
        }
        
      };

    }

  }

}
