#pragma once

#include <anns.hpp>
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

#include <base.hpp>

namespace anns
{

  namespace graph
  {

    template <typename data_t, float (*distance)(const data_t *, const data_t *, size_t)>
    class Vamana: public Base<data_t>
    {
      struct PHash
      {
        id_t operator()(const std::pair<float, id_t> &pr) const
        {
          return pr.second;
        }
      };

    protected:
      size_t R_{0}; // Graph degree limit
      size_t Lc_{0};
      float alpha_{.0};
      id_t enterpoint_node_{MAGIC_ID};
      std::vector<std::vector<id_t>> neighbors_;
      std::vector<std::unique_ptr<std::mutex>> link_list_locks_;
      std::mutex global_;

    public:
    __USE_BASE__

      Vamana(size_t R, size_t Lc, float alpha) noexcept : R_(R), Lc_(Lc), alpha_(alpha), enterpoint_node_(MAGIC_ID) {}

      Vamana(const DataSet<data_t> &base, const std::string &filename) noexcept
      {
        base_ = base;
        std::ifstream in(filename, std::ios::binary);
        in.read(reinterpret_cast<char *>(&R_), sizeof(R_));
        in.read(reinterpret_cast<char *>(&Lc_), sizeof(Lc_));
        in.read(reinterpret_cast<char *>(&alpha_), sizeof(alpha_));
        in.read(reinterpret_cast<char *>(&enterpoint_node_), sizeof(enterpoint_node_));
        neighbors_.resize(base_.num_);
        for (auto &ll : neighbors_)
        {
          size_t n;
          in.read(reinterpret_cast<char *>(&n), sizeof(n));
          ll.resize(n);
          char *buffer = reinterpret_cast<char *>(ll.data());
          in.read(buffer, n * sizeof(id_t));
        }
        link_list_locks_.resize(base_.num_);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](std::unique_ptr<std::mutex> &lock)
                      { lock = std::make_unique<std::mutex>(); });
      }

      void save(const std::string &filename) const noexcept override
      {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char *>(&base_.num_), sizeof(base_.num_));
        out.write(reinterpret_cast<const char *>(&R_), sizeof(R_));
        out.write(reinterpret_cast<const char *>(&base_.dim_), sizeof(base_.dim_));
        out.write(reinterpret_cast<const char *>(&Lc_), sizeof(Lc_));
        out.write(reinterpret_cast<const char *>(&alpha_), sizeof(alpha_));
        out.write(reinterpret_cast<const char *>(&enterpoint_node_), sizeof(enterpoint_node_));
        for (const auto &ll : neighbors_)
        {
          size_t n = ll.size();
          const char *buffer = reinterpret_cast<const char *>(ll.data());
          out.write(reinterpret_cast<const char *>(&n), sizeof(n));
          out.write(buffer, n * sizeof(id_t));
        }
      }

      void build(const DataSet<data_t> &base) override
      {
        base_ = base;
        neighbors_.assign(base_.num_, std::vector<id_t>(R_, MAGIC_ID));
        link_list_locks_.resize(base_.num_);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](std::unique_ptr<std::mutex> &lock)
                      { lock = std::make_unique<std::mutex>(); });
        // Initialize graph index to a random R-regular directed graph
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (id_t id = 0; id < base_.num_; id++)
        {
          auto &neighbors = neighbors_[id];
          for (size_t i = 0; i < R_; i++)
          {
            id_t rid = id;
            while (rid == id)
            {
              rid = (id_t)(rand() % base_.num_);
            }
            neighbors[i] = rid;
          }
        }
        // Compute medoid of the raw dataset
        std::vector<long double> dim_sum(base_.dim_, .0);
        std::vector<data_t> medoid(base_.dim_, 0);
        auto dim_lock_list = std::make_unique<std::vector<std::mutex>>(base_.dim_);

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (id_t id = 0; id < base_.num_; id++)
        {
          const data_t *vec = base_[id];
          for (size_t i = 0; i < base_.dim_; i++)
          {
            std::unique_lock<std::mutex> lock(dim_lock_list->at(i));
            dim_sum[i] += vec[i];
          }
        } //
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < base_.dim_; i++)
        {
          medoid[i] = static_cast<data_t>(dim_sum[i] / base_.num_);
        }
        float nearest_dist = MAGIC_DIST;
        id_t nearest_node = MAGIC_ID;

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (id_t id = 0; id < base_.num_; id++)
        {
          float dist = distance(medoid.data(), base_[id], base_.dim_);
          std::unique_lock<std::mutex> lock(global_);
          if (dist < nearest_dist)
          {
            nearest_dist = dist;
            nearest_node = id;
          }
        }
        enterpoint_node_ = nearest_node;

        // Generate a random permutation sigma
        std::vector<id_t> sigma(base_.num_);
#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (id_t id = 0; id < base_.num_; id++)
        {
          sigma[id] = id;
        }
        std::random_shuffle(sigma.begin(), sigma.end());

        // building pass begin
        auto pass = [&](float beta)
        {
#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
          for (size_t i = 0; i < base_.num_; i++)
          {
            id_t cur_id = sigma[i];
            auto top_candidates = search_base_layer(base_[cur_id], R_, Lc_);
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
            robust_prune(cur_id, beta, top_candidates);
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
                  temp_cand_set.emplace(-distance(base_[neij], base_[cur_id], base_.dim_), cur_id);
                  robust_prune(neij, beta, temp_cand_set);
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

      void search(const DataSet<data_t> &query, size_t k, size_t ef, matrix_id_t &knn, matrix_di_t &dists) override
      {

        size_t nq = query.num_;
        knn.clear();
        dists.clear();
        knn.resize(nq);
        dists.resize(nq);

#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads_)
        for (size_t i = 0; i < nq; i++)
        {
          const data_t *q = query[i];
          auto &vid = knn[i];
          auto &dist = dists[i];
          auto r = search_base_layer(q, k, ef);
          while (r.size())
          {
            const auto &tt = r.top();
            vid.emplace_back(tt.second);
            dist.emplace_back(tt.first);
            r.pop();
          }
        }
      }

      size_t index_size() const noexcept override
      {
        size_t sz = 0;
        for (const auto &ll : neighbors_)
        {
          sz += ll.size() * sizeof(id_t);
        }
        return sz;
      }

    protected:
      /// @brief Prune function
      /// @tparam data_t
      /// @param node_id
      /// @param alpha
      /// @param candidates a minheap
      void robust_prune(id_t node_id, float alpha, std::priority_queue<std::pair<float, id_t>> &candidates)
      {
        assert(alpha >= 1);

        // Ps: It will make a dead-lock if locked here, so make sure the code have locked the link-list of
        // the pruning node outside of the function `robust_prune` in caller
        const data_t *data_node = base_[node_id];
        auto &neighbors = neighbors_[node_id];
        for (id_t nei : neighbors)
        {
          candidates.emplace(-distance(base_[nei], data_node, base_.dim_), nei);
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
          const data_t *data_pstar = base_[pstar];
          std::priority_queue<std::pair<float, id_t>> temp;
          while (candidates.size())
          {
            auto [d, id] = candidates.top();
            candidates.pop();
            if (alpha * distance(data_pstar, base_[id], base_.dim_) <= -d)
              continue;
            temp.emplace(d, id);
          }
          candidates = std::move(temp);
        }
      }

      /// @brief search the base layer (User call this funtion to do single query).
      /// @param data_point
      /// @param k
      /// @param ef
      /// @return a maxheap containing the knn results
      std::priority_queue<std::pair<float, id_t>>
      search_base_layer(const data_t *data_point, size_t k, size_t ef)
      {
        std::vector<bool> mass_visited(base_.num_, false);
        std::priority_queue<std::pair<float, id_t>> top_candidates;
        std::priority_queue<std::pair<float, id_t>> candidate_set;
        size_t comparison = 0;
        float dist = distance(data_point, base_[enterpoint_node_], base_.dim_);
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
              float dd = distance(data_point, base_[neighbor_id], base_.dim_);
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
    };

  }

}
