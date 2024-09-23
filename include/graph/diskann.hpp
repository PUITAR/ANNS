#pragma once


#include <vector_ops.hpp>
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

    template <typename vdim_t>
    class DiskANN
    {
    public:
      size_t cur_element_count_{0};
      size_t R_{0}; // Graph degree limit
      size_t D_{0}; // vector dimension
      size_t Lc_{0};
      size_t Ld_{0}; // Delete vector batch size
      float alpha_{.0};
      id_t enterpoint_node_{0};
      std::unique_ptr<std::vector<const vdim_t *>> vector_data_{nullptr};
      std::unique_ptr<std::vector<std::vector<id_t>>> neighbors_{nullptr};
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
      std::priority_queue<id_t> free_ids_;
      std::vector<id_t> delete_list_;

      DiskANN(
          size_t D,
          size_t R,
          size_t Lc,
          float alpha,
          size_t Ld = 128,
          int random_seed = 123) : D_(D), R_(R), Lc_(Lc), alpha_(alpha), Ld_(Ld), random_seed_(random_seed)
      {
        vector_data_ = std::make_unique<std::vector<const vdim_t *>>();
        neighbors_ = std::make_unique<std::vector<std::vector<id_t>>>();
        cur_element_count_ = 0;
        enterpoint_node_ = -1;
      }

      inline const vdim_t *
      GetDataByInternalID(id_t id) const
      {
        return vector_data_->at(id);
      }

      inline void
      WriteDataByInternalID(id_t id, const vdim_t *data_point)
      {
        vector_data_->at(id) = data_point;
      }

      std::vector<id_t> &
      GetLinkByInternalID(id_t id) const
      {
        return neighbors_->at(id);
      }

      size_t GetNumThreads()
      {
        return num_threads_;
      }

      void SetNumThreads(size_t num_threads)
      {
        num_threads_ = num_threads;
      }

      /// @brief Search the base layer.
      /// @param data_point
      /// @param k
      /// @param ef
      /// @return a maxheap containing the knn results
      std::priority_queue<std::pair<float, id_t>>
      SearchBaseLayer(const vdim_t *data_point, size_t k, size_t ef)
      {
        auto mass_visited = std::make_unique<std::vector<bool>>(cur_element_count_, false);

        std::priority_queue<std::pair<float, id_t>> top_candidates;
        std::priority_queue<std::pair<float, id_t>> candidate_set;

        size_t comparison = 0;

        float dist = vec_L2sqr(data_point, GetDataByInternalID(enterpoint_node_), D_);
        comparison++;

        top_candidates.emplace(dist, enterpoint_node_); // max heap
        candidate_set.emplace(-dist, enterpoint_node_); // min heap
        mass_visited->at(enterpoint_node_) = true;

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
          const auto &neighbors = GetLinkByInternalID(curr_node_id);

          for (id_t neighbor_id : neighbors)
          {
            if (!mass_visited->at(neighbor_id))
            {
              mass_visited->at(neighbor_id) = true;

              float dd = vec_L2sqr(data_point, GetDataByInternalID(neighbor_id), D_);
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
      /// @tparam vdim_t
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
        const vdim_t *data_node = GetDataByInternalID(node_id);
        auto &neighbors = GetLinkByInternalID(node_id);
        for (id_t nei : neighbors)
        {
          if (!std::binary_search(delete_list_.begin(), delete_list_.end(), nei))
          {
            candidates.emplace(-vec_L2sqr(GetDataByInternalID(nei), data_node, D_), nei);
          }
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
          const vdim_t *data_pstar = GetDataByInternalID(pstar);
          std::priority_queue<std::pair<float, id_t>> temp;
          while (candidates.size())
          {
            auto [d, id] = candidates.top();
            candidates.pop();
            if (alpha * vec_L2sqr(data_pstar, GetDataByInternalID(id), D_) <= -d)
              continue;
            temp.emplace(d, id);
          }
          candidates = std::move(temp);
        }
      }

      void BuildIndex(const std::vector<vdim_t> &raw_data)
      {
        const size_t num_points = raw_data.size() / D_;
        cur_element_count_ = num_points;

        vector_data_ = std::make_unique<std::vector<const vdim_t *>>(num_points);
        neighbors_ = std::make_unique<std::vector<std::vector<id_t>>>(num_points, std::vector<id_t>(R_, -1));
        link_list_locks_.resize(num_points);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](std::unique_ptr<std::mutex> &lock)
                      { lock = std::make_unique<std::mutex>(); });
        for (id_t id = 0; id < num_points; id++)
        {
          WriteDataByInternalID(id, raw_data.data() + id * D_);
        }

        // Initialize graph index to a random R-regular directed graph
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          auto &neighbors = GetLinkByInternalID(id);
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
        std::vector<vdim_t> medoid(D_, 0);
        auto dim_lock_list = std::make_unique<std::vector<std::mutex>>(D_);

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          const vdim_t *vec = raw_data.data() + id * D_;
          for (size_t i = 0; i < D_; i++)
          {
            std::unique_lock<std::mutex> lock(dim_lock_list->at(i));
            dim_sum[i] += vec[i];
          }
        } //
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < D_; i++)
        {
          medoid[i] = static_cast<vdim_t>(dim_sum[i] / num_points);
        }
        float nearest_dist = std::numeric_limits<float>::max();
        id_t nearest_node = -1;

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          float dist = vec_L2sqr(medoid.data(), raw_data.data() + id * D_, D_);
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
            auto top_candidates = SearchBaseLayer(GetDataByInternalID(cur_id), R_, Lc_); // output is a maxheap containing results
            {                                                                            // transform to minheap
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
            auto &neighbors = GetLinkByInternalID(cur_id);
            std::vector<id_t> neighbors_copy(neighbors);
            lock.unlock();

            for (id_t neij : neighbors_copy)
            {
              std::unique_lock<std::mutex> lock_neij(*link_list_locks_[neij]);
              auto &neighbors_other = GetLinkByInternalID(neij);
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
                  temp_cand_set.emplace(-vec_L2sqr(GetDataByInternalID(neij), GetDataByInternalID(cur_id), D_), cur_id);
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

      void BuildIndex(const std::vector<const vdim_t *> &raw_data)
      {
        const size_t num_points = raw_data.size() / D_;
        cur_element_count_ = num_points;

        vector_data_ = std::make_unique<std::vector<const vdim_t *>>(raw_data);
        neighbors_ = std::make_unique<std::vector<std::vector<id_t>>>(num_points, std::vector<id_t>(R_, -1));
        link_list_locks_.resize(num_points);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](std::unique_ptr<std::mutex> &lock)
                      { lock = std::make_unique<std::mutex>(); });

        // Initialize graph index to a random R-regular directed graph
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          auto &neighbors = GetLinkByInternalID(id);
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
        std::vector<vdim_t> medoid(D_, 0);
        auto dim_lock_list = std::make_unique<std::vector<std::mutex>>(D_);

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          const vdim_t *vec = raw_data[id];
          for (size_t i = 0; i < D_; i++)
          {
            std::unique_lock<std::mutex> lock(dim_lock_list->at(i));
            dim_sum[i] += vec[i];
          }
        } //
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < D_; i++)
        {
          medoid[i] = static_cast<vdim_t>(dim_sum[i] / num_points);
        }
        float nearest_dist = std::numeric_limits<float>::max();
        id_t nearest_node = -1;

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          float dist = vec_L2sqr(medoid.data(), raw_data[id], D_);
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
            auto top_candidates = SearchBaseLayer(GetDataByInternalID(cur_id), R_, Lc_);
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
            auto &neighbors = GetLinkByInternalID(cur_id);
            std::vector<id_t> neighbors_copy(neighbors);
            lock.unlock();

            for (id_t neij : neighbors_copy)
            {
              std::unique_lock<std::mutex> lock_neij(*link_list_locks_[neij]);
              auto &neighbors_other = GetLinkByInternalID(neij);
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
                  temp_cand_set.emplace(-vec_L2sqr(GetDataByInternalID(neij), GetDataByInternalID(cur_id), D_), cur_id);
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
          const std::vector<std::vector<vdim_t>> &queries,
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

        std::sort(delete_list_.begin(), delete_list_.end()); // for binary search

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
            if (!std::binary_search(delete_list_.begin(), delete_list_.end(), tt.second))
            {
              vid.emplace_back(tt.second);
              dist.emplace_back(tt.first);
            }
            r.pop();
          }
        }
      }

      size_t GetComparisonAndClear()
      {
        return comparison_.exchange(0);
      }

      size_t IndexSize() const
      {
        size_t sz = 0;
        for (const auto &ll : *neighbors_)
        {
          sz += ll.size() * sizeof(id_t);
        }
        return sz;
      }

      id_t GetClosestPoint(const vdim_t *data_point)
      {
        if (cur_element_count_ == 0)
        {
          throw std::runtime_error("empty graph");
        }
        size_t comparison = 0;
        id_t wander = enterpoint_node_;
        float dist = vec_L2sqr(data_point, GetDataByInternalID(wander), D_);
        comparison++;
        bool moving = true;
        while (moving)
        {
          moving = false;
          auto &adj = GetLinkByInternalID(wander);
          size_t n = adj.size();
          for (size_t i = 0; i < n; i++)
          {
            id_t cand = adj[i];
            float d = vec_L2sqr(data_point, GetDataByInternalID(cand), D_);
            if (d < dist)
            {
              wander = cand;
              dist = d;
              moving = true;
            }
          }
          comparison += n;
        }
        comparison_.fetch_add(comparison);
        return wander;
      }

      // This method support insert vectors in the way of parallel, however with different sequence of insertion,
      // the result variant graph may be different.
      void Insert(const vdim_t *data)
      {
        // std::cout << "Inserting a new vector" << std::endl;
        id_t cur_id = AllocFreeVectorSpace(data); // return a free vector id
        // std::cout << "cur_id " << cur_id << std::endl;
        auto candidates = Search(data, 1, Lc_);
        std::unique_lock<std::mutex> lock(*link_list_locks_[cur_id]);
        RobustPrune(cur_id, alpha_, candidates);
        auto &neighbors = GetLinkByInternalID(cur_id);
        size_t num_neighbors = neighbors.size();
        std::vector<id_t> neighbors_copy(neighbors);
        lock.unlock();

        for (size_t j = 0; j < num_neighbors; j++)
        {
          id_t neij = neighbors_copy[j];
          std::unique_lock<std::mutex> lock_neij(*link_list_locks_[neij]);
          auto &neighbors_other = GetLinkByInternalID(neij);
          size_t num_neighbors_other = neighbors_other.size();
          bool find_cur_id = false;
          for (size_t k = 0; k < num_neighbors_other; k++)
          {
            if (cur_id == neighbors_other[k])
            {
              find_cur_id = true;
              break;
            }
          }

          if (!find_cur_id)
          {
            if (num_neighbors_other == R_)
            {
              std::priority_queue<std::pair<float, id_t>> temp_cand_set;
              temp_cand_set.emplace(-vec_L2sqr(GetDataByInternalID(neij), GetDataByInternalID(cur_id), D_), cur_id);
              RobustPrune(neij, alpha_, temp_cand_set);
            }
            else if (num_neighbors_other < R_)
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

      id_t AllocFreeVectorSpace(const vdim_t *data)
      {
        std::unique_lock<std::mutex> glock(global_);
        // std::cout << "free ids size " << free_ids_.size() << std::endl;
        if (free_ids_.size())
        {
          id_t id = -free_ids_.top();
          free_ids_.pop();
          vector_data_->at(id) = data;
          return id;
        }
        vector_data_->emplace_back(data);
        neighbors_->emplace_back(std::vector<id_t>());
        link_list_locks_.emplace_back(std::unique_ptr<std::mutex>(new std::mutex));
        return cur_element_count_++;
      }

      // User should not call this function in any parallel operation of deletion
      void Delete(id_t id)
      {
        // std::cout << "delete " << id << std::endl;
        if (neighbors_->at(id).size() == 0)
          return;

        while (id == enterpoint_node_ || neighbors_->at(enterpoint_node_).size() == 0)
          enterpoint_node_++;
        delete_list_.emplace_back(id);
        free_ids_.push(-id); // push the id into the free list
        if (delete_list_.size() >= Ld_)
        {
          ConsolidateDelete();
        }
      }

      // User should not call this function in any parallel operation of deletion
      void ConsolidateDelete()
      {
        // std::cout << "consolidate delete" << std::endl;
        std::sort(delete_list_.begin(), delete_list_.end());

#pragma omp parallel for num_threads(num_threads_)
        for (id_t id = 0; id < cur_element_count_; id++)
        {
          if (!std::binary_search(delete_list_.begin(), delete_list_.end(), id))
          {
            bool adjust = 0;
            const auto &neighbors = GetLinkByInternalID(id);
            for (id_t nei : neighbors)
            {
              if (std::binary_search(delete_list_.begin(), delete_list_.end(), nei))
              {
                adjust = 1;
                break;
              }
            }
            if (!adjust) // If there is no deleted nodes in the neighbors, we do not need to adjust the adjacency list
              continue;

            std::priority_queue<std::pair<float, id_t>> candidates;
            const vdim_t *vec = GetDataByInternalID(id);
            for (id_t did : delete_list_)
            {
              const auto &neighbors_other = GetLinkByInternalID(did);
              for (id_t nei : neighbors_other)
              {
                if (!std::binary_search(delete_list_.begin(), delete_list_.end(), nei))
                {
                  candidates.emplace(-vec_L2sqr(vec, GetDataByInternalID(nei), D_), nei);
                }
              }
            }
            RobustPrune(id, alpha_, candidates);
          }
        }
        // Clear all delete node
        for (id_t id : delete_list_)
        {
          neighbors_->at(id).clear();
        }
        delete_list_.clear();
      }
    };

  } // namespace graph

} // namespace index
