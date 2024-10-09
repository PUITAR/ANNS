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
#include <atomic>
#include <utils/binary.hpp>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <omp.h>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

namespace anns
{

  namespace graph
  {

    template <typename data_t, float (*distance)(const data_t *, const data_t *, size_t)>
    class HNSW
    {

    public:
#define MAGIC_VEC_ID std::numeric_limits<id_t>::max()

      size_t cur_element_count_{0};
      size_t D_{0}; // vector dimensions
      size_t Mmax_{0};            // maximum number of connections for each element per layer
      size_t Mmax0_{0};           // maximum number of connections for each element in layer0
      size_t ef_construction_{0}; // usually been set to 128
      double mult_{0.0};
      double rev_size_{0.0};
      int max_level_{0};
      id_t enterpoint_node_{MAGIC_VEC_ID};
      int random_seed_{100};

      std::vector<int> element_levels_; // keeps level of each element
      std::vector<std::vector<std::vector<id_t>>> link_lists_;
      std::default_random_engine level_generator_;
      std::vector<std::unique_ptr<std::mutex>> link_list_locks_;
      std::vector<const data_t*> data_memory_; // vector data start pointer of memory.

      std::mutex global_;
      size_t num_threads_{1};
      std::atomic<size_t> comparison_{0};

      HNSW(size_t D, size_t M, size_t ef_construction, size_t random_seed = 100) noexcept: 
        D_(D), Mmax_(M), Mmax0_(2 * M), ef_construction_(std::max(ef_construction, M)), random_seed_(random_seed), mult_(1 / log(1.0 * Mmax_)), rev_size_(1.0 / mult_)
      {
        level_generator_.seed(random_seed);
      }

      HNSW(const std::vector<data_t>& base, const std::string& filename) noexcept
      {
        std::ifstream in(filename, std::ios::binary);
        in.read(reinterpret_cast<char*>(&cur_element_count_), sizeof(cur_element_count_));
        in.read(reinterpret_cast<char*>(&D_), sizeof(D_));
        in.read(reinterpret_cast<char*>(&Mmax_), sizeof(Mmax_));
        in.read(reinterpret_cast<char*>(&Mmax0_), sizeof(Mmax0_));
        in.read(reinterpret_cast<char*>(&ef_construction_), sizeof(ef_construction_));
        in.read(reinterpret_cast<char*>(&mult_), sizeof(mult_));
        in.read(reinterpret_cast<char*>(&rev_size_), sizeof(rev_size_));
        in.read(reinterpret_cast<char*>(&max_level_), sizeof(max_level_));
        in.read(reinterpret_cast<char*>(&enterpoint_node_), sizeof(enterpoint_node_));
        in.read(reinterpret_cast<char*>(&random_seed_), sizeof(random_seed_));
        element_levels_.resize(cur_element_count_);
        in.read(reinterpret_cast<char*>(element_levels_.data()), cur_element_count_ * sizeof(int));
        link_lists_.resize(cur_element_count_);
        for (id_t id = 0; id < cur_element_count_; id++)
        {
          auto& ll = link_lists_[id];
          ll.resize(element_levels_[id] + 1);
          for (auto& l: ll)
          {
            size_t n;
            in.read(reinterpret_cast<char*>(&n), sizeof(size_t));
            l.resize(n);
            in.read(reinterpret_cast<char*>(l.data()), n * sizeof(id_t));
          }
        }
        level_generator_.seed(random_seed_);
        link_list_locks_.resize(cur_element_count_);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](std::unique_ptr<std::mutex>& lock) 
          { lock = std::make_unique<std::mutex>(); });
        data_memory_.reserve(cur_element_count_);
        for (size_t i = 0; i < cur_element_count_; i++)
        {
          data_memory_.emplace_back(base.data() + i * D_);
        }
      }

      void save(const std::string& filename) const noexcept
      {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&cur_element_count_), sizeof(cur_element_count_));
        out.write(reinterpret_cast<const char*>(&D_), sizeof(D_));
        out.write(reinterpret_cast<const char*>(&Mmax_), sizeof(Mmax_));
        out.write(reinterpret_cast<const char*>(&Mmax0_), sizeof(Mmax0_));
        out.write(reinterpret_cast<const char*>(&ef_construction_), sizeof(ef_construction_));
        out.write(reinterpret_cast<const char*>(&mult_), sizeof(mult_));
        out.write(reinterpret_cast<const char*>(&rev_size_), sizeof(rev_size_));
        out.write(reinterpret_cast<const char*>(&max_level_), sizeof(max_level_));
        out.write(reinterpret_cast<const char*>(&enterpoint_node_), sizeof(enterpoint_node_));
        out.write(reinterpret_cast<const char*>(&random_seed_), sizeof(random_seed_));
        const char* buffer = reinterpret_cast<const char*>(element_levels_.data());
        out.write(buffer, element_levels_.size() * sizeof(int));
        for (const auto& ll: link_lists_)
        {
          for (const auto& l: ll)
          {
            size_t n = l.size();
            const char* buffer = reinterpret_cast<const char*>(l.data());
            out.write(reinterpret_cast<const char*>(&n), sizeof(n));
            out.write(buffer, sizeof(id_t) * n);
          }
        }
      }
      
      /// @brief  Add a point to the graph [User should not call this function directly]
      /// @param data_point 
      void build_point(id_t cur_id, const data_t *data_point)
      {
        // Write the data point
        data_memory_[cur_id] = data_point;

        // alloc memory for the link lists
        std::unique_lock<std::mutex> lock_el(*link_list_locks_[cur_id]);
        int cur_level = get_random_level(mult_);
        for (int lev = 0; lev <= cur_level; lev++)
        {
          link_lists_[cur_id].emplace_back(std::vector<id_t>());
        }
        element_levels_[cur_id] = cur_level;

        std::unique_lock<std::mutex> temp_lock(global_);
        int max_level_copy = max_level_;
        id_t cur_obj = enterpoint_node_;
        id_t enterpoint_node_copy = enterpoint_node_;
        if (cur_level <= max_level_) temp_lock.unlock();

        if (enterpoint_node_copy != MAGIC_VEC_ID)
        { // not first element
          if (cur_level < max_level_copy)
          {
            // find the closet node in upper layers
            float cur_dist = distance(data_point, data_memory_[cur_obj], D_);
            for (int lev = max_level_copy; lev > cur_level; lev--)
            {
              bool changed = true;
              while (changed)
              {
                changed = false;
                std::unique_lock<std::mutex> wlock(*link_list_locks_[cur_obj]);
                const auto& neighbors = link_lists_[cur_obj][lev];
                size_t num_neighbors = neighbors.size();

                for (size_t i = 0; i < num_neighbors; i++)
                {
                  id_t cand = neighbors[i];
                  float d = distance(data_point, data_memory_[cand], D_);
                  if (d < cur_dist)
                  {
                    cur_dist = d;
                    cur_obj = cand;
                    changed = true;
                  }
                }
              }
            }
          }
          /// add edges to lower layers from the closest node
          for (int lev = std::min(cur_level, max_level_copy); lev >= 0; lev--)
          {
            auto top_candidates = search_base_layer(cur_obj, data_point, lev, ef_construction_);
            cur_obj = mutually_connect_new_element(data_point, cur_id, top_candidates, lev);
          }
        }
        else
        {
          // Do nothing for the first element
          enterpoint_node_ = cur_id;
          max_level_ = cur_level;
        }

        // Releasing lock for the maximum level
        if (cur_level > max_level_copy)
        {
          enterpoint_node_ = cur_id;
          max_level_ = cur_level;
        }
      }

      void build(const std::vector<data_t> &raw_data)
      {
        size_t num_points = raw_data.size() / D_;
        cur_element_count_ = num_points;

        data_memory_.resize(num_points);
        link_lists_.resize(num_points);
        link_list_locks_.resize(num_points);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](auto &lock){ lock = std::make_unique<std::mutex>(); });
        element_levels_.resize(num_points, 0);

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          build_point(id, raw_data.data() + id * D_);
        }
      }

      void build(const std::vector<const data_t *> &raw_data)
      {
        size_t num_points = raw_data.size();
        cur_element_count_ = num_points;

        data_memory_.resize(num_points);
        link_lists_.resize(num_points);
        link_list_locks_.resize(num_points);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](auto &lock){ lock = std::make_unique<std::mutex>(); });
        element_levels_.resize(num_points, 0);

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          build_point(id, raw_data[id]);
        }
      }

      std::priority_queue<std::pair<float, id_t>> search(const data_t *query_data, size_t k, size_t ef)
      {
        assert(ef >= k && "ef > k!");

        if (cur_element_count_ == 0)
          return std::priority_queue<std::pair<float, id_t>>();

        size_t comparison = 0;
        id_t cur_obj = enterpoint_node_;
        float cur_dist = distance(query_data, data_memory_[enterpoint_node_], D_);
        comparison++;

        for (int lev = element_levels_[enterpoint_node_]; lev > 0; lev--)
        {
          // find the closet node in upper layers
          bool changed = true;
          while (changed)
          {
            changed = false;
            const auto& neighbors = link_lists_[cur_obj][lev];
            size_t num_neighbors = neighbors.size();

            for (size_t i = 0; i < num_neighbors; i++)
            {
              id_t cand = neighbors[i];
              float d = distance(query_data, data_memory_[cand], D_);
              if (d < cur_dist)
              {
                cur_dist = d;
                cur_obj = cand;
                changed = true;
              }
            }
            comparison += num_neighbors;
          }
        }

        auto top_candidates = search_base_layer(cur_obj, query_data, 0, ef);

        while (top_candidates.size() > k)
        {
          top_candidates.pop();
        }

        comparison_.fetch_add(comparison);

        return top_candidates;
      }

      std::priority_queue<std::pair<float, id_t>> search(const data_t *query_data, size_t k, size_t ef, id_t ep)
      {
        assert(ef >= k && "ef > k!");

        if (cur_element_count_ == 0)
          return std::priority_queue<std::pair<float, id_t>>();

        size_t comparison = 0;

        id_t cur_obj = ep;
        float cur_dist = distance(query_data, data_memory_[enterpoint_node_], D_);
        comparison++;

        for (int lev = element_levels_[ep]; lev > 0; lev--)
        {
          // find the closet node in upper layers
          bool changed = true;
          while (changed)
          {
            changed = false;
            const auto& neighbors = link_lists_[cur_obj][lev];
            size_t num_neighbors = neighbors.size();

            for (size_t i = 0; i < num_neighbors; i++)
            {
              id_t cand = neighbors[i];
              float d = distance(query_data, data_memory_[cand], D_);
              if (d < cur_dist)
              {
                cur_dist = d;
                cur_obj = cand;
                changed = true;
              }
            }
            comparison += num_neighbors;
          }
        }

        auto top_candidates = search_base_layer(cur_obj, query_data, 0, ef);

        while (top_candidates.size() > k)
        {
          top_candidates.pop();
        }

        comparison_.fetch_add(comparison);

        return top_candidates;
      }

      void search(const std::vector<std::vector<data_t>> &queries, size_t k, size_t ef, std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists)
      {
        size_t nq = queries.size();
        vids.clear();
        dists.clear();
        vids.resize(nq);
        dists.resize(nq);

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < nq; i++)
        {
          const auto &query = queries[i];
          auto &vid = vids[i];
          auto &dist = dists[i];

          auto r = search(query.data(), k, ef);
          vid.reserve(r.size());
          dist.reserve(r.size());
          while (r.size())
          {
            const auto &te = r.top();
            vid.emplace_back(te.second);
            dist.emplace_back(te.first);
            r.pop();
          }
        }
      }

      size_t get_num_threads() const noexcept
      {
        return num_threads_;
      }

      void set_num_threads(size_t num_threads) noexcept
      {
        num_threads_ = num_threads;
      }

      size_t get_comparison_and_clear() noexcept
      {
        return comparison_.exchange(0);
      }

      size_t index_size() const noexcept
      {
        size_t sz = 0;
        for (const auto & ll: link_lists_) {
          for (const auto & l: ll) {
            sz += l.size() * sizeof(id_t);
          }
        }
        return sz;
      }

      /// @brief Connection new element and return next cloest element id
      /// @param data_point
      /// @param id
      /// @param top_candidates
      /// @param layer
      /// @return
      id_t mutually_connect_new_element(const data_t *data_point, id_t id, std::priority_queue<std::pair<float, id_t>> &top_candidates, int level)
      {
        size_t Mcurmax = level ? Mmax_ : Mmax0_;
        prune_neighbors(top_candidates, Mcurmax);

        auto& neighbors_cur = link_lists_[id][level];
        /// @brief Edge-slots check and Add neighbors for current vector
        {
          // lock only during the update
          // because during the addition the lock for cur_c is already acquired
          std::unique_lock<std::mutex> lock(*link_list_locks_[id], std::defer_lock);
          neighbors_cur.clear();
          assert(top_candidates.size() <= Mcurmax);
          neighbors_cur.reserve(top_candidates.size());
          
          while (top_candidates.size())
          {
            neighbors_cur.emplace_back(top_candidates.top().second);
            top_candidates.pop();
          }
        }

        id_t next_closet_entry_point = neighbors_cur.back();

        for (id_t sid: neighbors_cur)
        {
          std::unique_lock<std::mutex> lock(*link_list_locks_[sid]);

          auto& neighbors = link_lists_[sid][level];
          size_t sz_link_list_other = neighbors.size();

          if (sz_link_list_other > Mcurmax)
          {
            std::cout << sz_link_list_other << ">" << Mcurmax << std::endl;
            std::cerr << "Bad value of sz_link_list_other" << std::endl;
            exit(1);
          }
          if (sid == id)
          {
            std::cerr << "Trying to connect an element to itself" << std::endl;
            exit(1);
          }
          if (level > element_levels_[sid])
          {
            std::cerr << "Trying to make a link on a non-existent level" << std::endl;
            exit(1);
          }

          if (sz_link_list_other < Mcurmax)
          {
            neighbors.emplace_back(id);
          }
          else
          {
            // finding the "farest" element to replace it with the new one
            float d_max = distance(data_memory_[id], data_memory_[sid], D_);
            // Heuristic:
            std::priority_queue<std::pair<float, id_t>> candidates;
            candidates.emplace(d_max, id);

            for (size_t j = 0; j < sz_link_list_other; j++)
            {
              candidates.emplace(distance(data_memory_[neighbors[j]], data_memory_[sid], D_), neighbors[j]);
            }

            prune_neighbors(candidates, Mcurmax);
            // Copy neighbors and add edges
            neighbors.clear();
            neighbors.reserve(candidates.size());
            while (candidates.size())
            {
              neighbors.emplace_back(candidates.top().second);
              candidates.pop();
            }
          }
        }

        return next_closet_entry_point;
      }

      /// @brief Return max heap of the top NN elements
      /// @param top_candidates 
      /// @param NN 
      void prune_neighbors(std::priority_queue<std::pair<float, id_t>> &top_candidates, size_t NN)
      {
        if (top_candidates.size() < NN)
        {
          return;
        }

        std::priority_queue<std::pair<float, id_t>> queue_closest; // min heap
        std::vector<std::pair<float, id_t>> return_list;

        while (top_candidates.size())
        { // replace top_candidates with a min-heap, so that each poping can return the nearest neighbor.
          const auto &te = top_candidates.top();
          queue_closest.emplace(-te.first, te.second);
          top_candidates.pop();
        }

        while (queue_closest.size())
        {
          if (return_list.size() >= NN)
          {
            break;
          }

          const auto curen = queue_closest.top();
          float dist2query = -curen.first;
          const data_t *curenv = data_memory_[curen.second];
          queue_closest.pop();
          bool good = true;
          for (const auto &curen2 : return_list)
          {
            float dist2curenv2 = distance(data_memory_[curen2.second], curenv, D_);
            if (dist2curenv2 < dist2query)
            {
              good = false;
              break;
            }
          }
          if (good)
          {
            return_list.emplace_back(curen);
          }
        }

        for (const auto &elem : return_list)
        {
          top_candidates.emplace(-elem.first, elem.second);
        }
      }

      int get_random_level(double reverse_size)
      {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
      }

      /// @brief Return the topk nearest neighbors (max-heap) of a given data point on a certain level
      /// @param ep_id 
      /// @param data_point 
      /// @param level 
      /// @param ef 
      /// @return 
      std::priority_queue<std::pair<float, id_t>> search_base_layer(id_t ep_id, const data_t *data_point, int level, size_t ef)
      {
        size_t comparison = 0;
        std::vector<bool> mass_visited(cur_element_count_, false);
        std::priority_queue<std::pair<float, id_t>> top_candidates;
        std::priority_queue<std::pair<float, id_t>> candidate_set;
        float dist = distance(data_point, data_memory_[ep_id], D_);
        comparison++;
        top_candidates.emplace(dist, ep_id); // max heap
        candidate_set.emplace(-dist, ep_id); // min heap
        mass_visited[ep_id] = true;

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
          const auto& neighbors = link_lists_[curr_node_id][level];

          for (id_t neighbor_id: neighbors)
          {
            if (mass_visited[neighbor_id] == false)
            {
              mass_visited[neighbor_id] = true;

              float dist = distance(data_point, data_memory_[neighbor_id], D_);
              comparison++;

              /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
              if (top_candidates.top().first > dist || top_candidates.size() < ef)
              {
                candidate_set.emplace(-dist, neighbor_id);
                top_candidates.emplace(dist, neighbor_id);

                if (top_candidates.size() > ef) // give up farest result so far
                  top_candidates.pop();

                if (top_candidates.size())
                  low_bound = top_candidates.top().first;
              }
            }
          }
        }
        comparison_.fetch_add(comparison);
        return top_candidates;
      }

    };

  }; // namespace graph

}; // namespace index
