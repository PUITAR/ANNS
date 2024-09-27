#pragma once

#include <distance.hpp>
#include <random>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>
#include <queue>
#include <memory>
#include <mutex>
#include <algorithm>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

#include <atomic>
#include <omp.h>

namespace anns
{

  namespace graph
  {

    template <typename data_t, float (*distance)(const data_t *, const data_t *, size_t)>
    class Vamana
    {
    public:
      size_t cur_element_count_{0};
      size_t R_{0}; // Graph degree limit
      size_t D_{0}; // vector dimension
      size_t Lc_{0};
      float alpha_{.0};
      id_t enterpoint_node_{0};
      std::vector<const data_t *> vector_data_;
      std::vector<std::vector<id_t>> neighbors_;
      int random_seed_{123};
      // bool ready_{false};
      size_t num_threads_{1};
      std::mutex global_;
      std::vector<std::unique_ptr<std::mutex>> link_list_locks_;
      struct PHash
      {
        id_t operator()(const std::pair<float, id_t> &pr) const
        {
          return pr.second;
        }
      };
      std::atomic<size_t> comparison_{0};

      Vamana(size_t D, size_t R, size_t Lc, float alpha, int random_seed = 123) noexcept : D_(D), R_(R), Lc_(Lc), alpha_(alpha), random_seed_(random_seed), cur_element_count_(0), enterpoint_node_(-1) {}

      size_t GetNumThreads() noexcept
      {
        return num_threads_;
      }

      void SetNumThreads(size_t num_threads) noexcept
      {
        num_threads_ = num_threads;
      }

      /// @brief Search the base layer (User call this funtion to do single query).
      /// @param data_point
      /// @param k
      /// @param ef
      /// @return a maxheap containing the knn results
      std::priority_queue<std::pair<float, id_t>>
      SearchBaseLayer(const data_t *data_point, size_t k, size_t ef)
      {
        std::vector<bool> mass_visited(cur_element_count_, false);

        std::priority_queue<std::pair<float, id_t>> top_candidates;
        std::priority_queue<std::pair<float, id_t>> candidate_set;

        size_t comparison = 0;

        float dist = distance(data_point, vector_data_[enterpoint_node_], D_);
        comparison++;

        top_candidates.emplace(dist, enterpoint_node_); // max heap
        candidate_set.emplace(-dist, enterpoint_node_); // min heap
        mass_visited[enterpoint_node_] = true;

        /// @brief Branch and Bound Algorithm
        float low_bound = dist;
        while (candidate_set.size())
        {
          auto curr_el_pair = candidate_set.top();
          if (-curr_el_pair.first > low_bound && top_candidates.size() == ef)
            break;

          candidate_set.pop();
          id_t curr_node_id = curr_el_pair.second;

          std::unique_lock<std::mutex> lock(*link_list_locks_[curr_node_id]);
          const auto &neighbors = neighbors_[curr_node_id];

          for (id_t neighbor_id : neighbors)
          {
            if (!mass_visited[neighbor_id])
            {
              mass_visited[neighbor_id] = true;

              float dd = distance(data_point, vector_data_[neighbor_id], D_);
              comparison++;

              /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
              if (top_candidates.top().first > dd || top_candidates.size() < ef)
              {
                candidate_set.emplace(-dd, neighbor_id);
                top_candidates.emplace(dd, neighbor_id);

                if (top_candidates.size() > ef) // give up farest result so far
                  top_candidates.pop();

                if (top_candidates.size())
                  low_bound = top_candidates.top().first;
              }
            }
          }
        }

        while (top_candidates.size() > k)
        {
          top_candidates.pop();
        }

        comparison_.fetch_add(comparison);
        return top_candidates;
      }

      /// @brief Prune function
      /// @tparam data_t
      /// @param node_id
      /// @param alpha
      /// @param candidates a minheap
      void RobustPrune(
          id_t node_id,
          float alpha,
          std::priority_queue<std::pair<float, id_t>> &candidates)
      {
        assert(alpha >= 1);

        // Ps: It will make a dead-lock if locked here, so make sure the code have locked the link-list of
        // the pruning node outside of the function `RobustPrune` in caller
        const data_t *data_node = vector_data_[node_id];
        auto &neighbors = neighbors_[node_id];
        for (id_t nei : neighbors)
        {
          candidates.emplace(-distance(vector_data_[nei], data_node, D_), nei);
        }

        { // Remove all deduplicated nodes
          std::unordered_set<std::pair<float, id_t>, PHash> cand_set;
          while (candidates.size())
          {
            const auto &top = candidates.top();
            if (top.second != node_id)
            {
              cand_set.insert(top);
            }
            candidates.pop();
          }
          for (const auto &[d, id] : cand_set)
          {
            candidates.emplace(d, id);
          }
        }

        neighbors.clear();        // clear link list
        while (candidates.size()) // candidates is a minheap, which means that the distance in the candidatas are negtive
        {
          if (neighbors.size() >= R_)
            break;
          auto [pstar_dist, pstar] = candidates.top();
          candidates.pop();
          neighbors.emplace_back(pstar);
          const data_t *data_pstar = vector_data_[pstar];
          std::priority_queue<std::pair<float, id_t>> temp;
          while (candidates.size())
          {
            auto [d, id] = candidates.top();
            candidates.pop();
            if (alpha * distance(data_pstar, vector_data_[id], D_) <= -d)
              continue;
            temp.emplace(d, id);
          }
          candidates = std::move(temp);
        }
      }

      void BuildIndex(const std::vector<data_t> &raw_data)
      {
        const size_t num_points = raw_data.size() / D_;
        cur_element_count_ = num_points;

        vector_data_.resize(num_points);
        neighbors_.assign(num_points, std::vector<id_t>(R_, -1));
        link_list_locks_.resize(num_points);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](std::unique_ptr<std::mutex> &lock)
                      { lock = std::make_unique<std::mutex>(); });
        for (id_t id = 0; id < num_points; id++)
        {
          vector_data_[id] = raw_data.data() + id * D_;
        }

        // Initialize graph index to a random R-regular directed graph
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          auto &neighbors = neighbors_[id];
          for (size_t i = 0; i < R_; i++)
          {
            id_t rid = id;
            while (rid == id)
            {
              rid = (id_t)(rand() % num_points);
            }
            neighbors[i] = rid;
          }
        }
        // Compute medoid of the raw dataset
        std::vector<long double> dim_sum(D_, .0);
        std::vector<data_t> medoid(D_, 0);
        auto dim_lock_list = std::make_unique<std::vector<std::mutex>>(D_);

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          const data_t *vec = raw_data.data() + id * D_;
          for (size_t i = 0; i < D_; i++)
          {
            std::unique_lock<std::mutex> lock(dim_lock_list->at(i));
            dim_sum[i] += vec[i];
          }
        } //
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < D_; i++)
        {
          medoid[i] = static_cast<data_t>(dim_sum[i] / num_points);
        }
        float nearest_dist = std::numeric_limits<float>::max();
        id_t nearest_node = -1;

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          float dist = distance(medoid.data(), raw_data.data() + id * D_, D_);
          std::unique_lock<std::mutex> lock(global_);
          if (dist < nearest_dist)
          {
            nearest_dist = dist;
            nearest_node = id;
          }
        }
        enterpoint_node_ = nearest_node;

        // Generate a random permutation sigma
        std::vector<id_t> sigma(num_points);
#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          sigma[id] = id;
        }
        std::random_shuffle(sigma.begin(), sigma.end());

        // Building pass begin
        auto pass = [&](float beta)
        {
#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
          for (size_t i = 0; i < num_points; i++)
          {
            id_t cur_id = sigma[i];
            auto top_candidates = SearchBaseLayer(vector_data_[cur_id], R_, Lc_); // output is a maxheap containing results
            {                                                                     // transform to minheap
              std::priority_queue<std::pair<float, id_t>> temp;
              while (top_candidates.size())
              {
                const auto &[d, id] = top_candidates.top();
                temp.emplace(-d, id);
                top_candidates.pop();
              }
              top_candidates = std::move(temp);
            }
            std::unique_lock<std::mutex> lock(*link_list_locks_[cur_id]);
            RobustPrune(cur_id, beta, top_candidates);
            auto &neighbors = neighbors_[cur_id];
            std::vector<id_t> neighbors_copy(neighbors);
            lock.unlock();

            for (id_t neij : neighbors_copy)
            {
              std::unique_lock<std::mutex> lock_neij(*link_list_locks_[neij]);
              auto &neighbors_other = neighbors_[neij];
              bool find_cur_id = false;
              for (id_t idno : neighbors_other)
              {
                if (cur_id == idno)
                {
                  find_cur_id = true;
                  break;
                }
              }

              if (!find_cur_id)
              {
                if (neighbors_other.size() == R_)
                {
                  std::priority_queue<std::pair<float, id_t>> temp_cand_set;
                  temp_cand_set.emplace(-distance(vector_data_[neij], vector_data_[cur_id], D_), cur_id);
                  RobustPrune(neij, beta, temp_cand_set);
                }
                else if (neighbors_other.size() < R_)
                {
                  neighbors_other.emplace_back(cur_id);
                }
                else
                {
                  throw std::runtime_error("adjency overflow");
                }
              }
            }
          }
        };

        /// First pass with alpha = 1.0
        pass(1.0);
        /// Second pass with user-defined alpha
        pass(alpha_);
      }

      void BuildIndex(const std::vector<const data_t *> &raw_data)
      {
        const size_t num_points = raw_data.size() / D_;
        cur_element_count_ = num_points;

        vector_data_.resize(num_points);
        neighbors_.assign(num_points, std::vector<id_t>(R_, -1));
        link_list_locks_.resize(num_points);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](std::unique_ptr<std::mutex> &lock)
                      { lock = std::make_unique<std::mutex>(); });

        // Initialize graph index to a random R-regular directed graph
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          auto &neighbors = neighbors_[id];
          for (size_t i = 0; i < R_; i++)
          {
            id_t rid = id;
            while (rid == id)
            {
              rid = (id_t)(rand() % num_points);
            }
            neighbors[i] = rid;
          }
        }
        // Compute medoid of the raw dataset
        std::vector<long double> dim_sum(D_, .0);
        std::vector<data_t> medoid(D_, 0);
        auto dim_lock_list = std::make_unique<std::vector<std::mutex>>(D_);

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          const data_t *vec = raw_data[id];
          for (size_t i = 0; i < D_; i++)
          {
            std::unique_lock<std::mutex> lock(dim_lock_list->at(i));
            dim_sum[i] += vec[i];
          }
        } //
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < D_; i++)
        {
          medoid[i] = static_cast<data_t>(dim_sum[i] / num_points);
        }
        float nearest_dist = std::numeric_limits<float>::max();
        id_t nearest_node = -1;

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          float dist = distance(medoid.data(), raw_data[id], D_);
          std::unique_lock<std::mutex> lock(global_);
          if (dist < nearest_dist)
          {
            nearest_dist = dist;
            nearest_node = id;
          }
        }
        enterpoint_node_ = nearest_node;

        // Generate a random permutation sigma
        std::vector<id_t> sigma(num_points);
#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          sigma[id] = id;
        }
        std::random_shuffle(sigma.begin(), sigma.end());

        // Building pass begin
        auto pass = [&](float beta)
        {
#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
          for (size_t i = 0; i < num_points; i++)
          {
            id_t cur_id = sigma[i];
            auto top_candidates = SearchBaseLayer(vector_data_[cur_id], R_, Lc_);
            { // transform to minheap
              std::priority_queue<std::pair<float, id_t>> temp;
              while (top_candidates.size())
              {
                const auto &[d, id] = top_candidates.top();
                temp.emplace(-d, id);
                top_candidates.pop();
              }
              top_candidates = std::move(temp);
            }

            std::unique_lock<std::mutex> lock(*link_list_locks_[cur_id]);
            RobustPrune(cur_id, beta, top_candidates);
            auto &neighbors = neighbors_[cur_id];
            std::vector<id_t> neighbors_copy(neighbors);
            lock.unlock();

            for (id_t neij : neighbors_copy)
            {
              std::unique_lock<std::mutex> lock_neij(*link_list_locks_[neij]);
              auto &neighbors_other = neighbors_[neij];
              bool find_cur_id = false;
              for (id_t idno : neighbors_other)
              {
                if (cur_id == idno)
                {
                  find_cur_id = true;
                  break;
                }
              }

              if (!find_cur_id)
              {
                if (neighbors_other.size() == R_)
                {
                  std::priority_queue<std::pair<float, id_t>> temp_cand_set;
                  temp_cand_set.emplace(-distance(vector_data_[neij], vector_data_[cur_id], D_), cur_id);
                  RobustPrune(neij, beta, temp_cand_set);
                }
                else if (neighbors_other.size() < R_)
                {
                  neighbors_other.emplace_back(cur_id);
                }
                else
                {
                  throw std::runtime_error("adjency overflow");
                }
              }
            }
          }
        };

        /// First pass with alpha = 1.0
        pass(1.0);
        // std::cout << "First pass done" << std::endl;
        /// Second pass with user-defined alpha
        pass(alpha_);
        // std::cout << "Second pass done" << std::endl;
      }

      void Search(
          const std::vector<std::vector<data_t>> &queries,
          size_t k,
          size_t ef,
          std::vector<std::vector<id_t>> &knn,
          std::vector<std::vector<float>> &dists)
      {

        size_t nq = queries.size();
        knn.clear();
        dists.clear();
        knn.resize(nq);
        dists.resize(nq);

#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads_)
        for (size_t i = 0; i < nq; i++)
        {
          const auto &query = queries[i];
          auto &vid = knn[i];
          auto &dist = dists[i];
          auto r = SearchBaseLayer(query.data(), k, ef);
          while (r.size())
          {
            const auto &tt = r.top();
            vid.emplace_back(tt.second);
            dist.emplace_back(tt.first);
            r.pop();
          }
        }
      }

      size_t GetComparisonAndClear() noexcept
      {
        return comparison_.exchange(0);
      }

      size_t IndexSize() const noexcept
      {
        size_t sz = 0;
        for (const auto &ll : neighbors_)
        {
          sz += ll.size() * sizeof(id_t);
        }
        return sz;
      }
      
    };

  } // namespace graph

} // namespace index
