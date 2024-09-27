#pragma once

#include <distance.hpp>
#include <random>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_set>
#include <map>
#include <cmath>
#include <queue>
#include <memory>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

#include <atomic>
#include <mutex>
#include <omp.h>
#include <algorithm>
#include <memory>

#include <utils/binary.hpp>

namespace anns
{

  namespace graph
  {

    class DisjointSet
    {
    public:
      DisjointSet(size_t size)
      {
        parent.resize(size);
        for (size_t i = 0; i < size; ++i)
          parent[i] = i;
      }

      id_t Find(id_t x)
      {
        if (parent[x] != x)
          parent[x] = Find(parent[x]);
        return parent[x];
      }

      void UnionSet(id_t x, id_t y)
      {
        parent[Find(x)] = Find(y);
      }

      std::vector<id_t> parent;
    };

    struct Edge
    {
      id_t src;
      id_t dst;
      float weight;

      bool operator<(const Edge &other) const
      {
        return this->weight < other.weight;
      }
    };

    template <typename data_t, float (*distance)(const data_t *, const data_t *, size_t)>
    class HCNNG
    {
    public:
      size_t num_threads_{1};
      std::atomic<size_t> comparison_{0};
      std::vector<const data_t *> data_memory_;
      
      size_t cur_element_count_{0};
      size_t D_;
      size_t num_random_clusters_{0};
      size_t min_size_clusters_{0};
      size_t max_mst_degree_{0};
      std::vector<std::vector<id_t>> adj_memory_;

      std::vector<std::unique_ptr<std::mutex>> link_list_locks_;

      HCNNG(size_t D, size_t max_elements, size_t num_random_clusters, size_t min_size_clusters, size_t max_mst_degree) noexcept : 
        D_(D), cur_element_count_(0), min_size_clusters_(min_size_clusters), max_mst_degree_(max_mst_degree), num_random_clusters_(num_random_clusters) {}

      HCNNG(const std::vector<data_t>& base, const std::string& filename) noexcept
      {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open())
        {
          throw std::runtime_error("Cannot open file for reading");
        }
        in.read(reinterpret_cast<char *>(&cur_element_count_), sizeof(cur_element_count_));
        in.read(reinterpret_cast<char *>(&D_), sizeof(D_));
        in.read(reinterpret_cast<char *>(&num_random_clusters_), sizeof(num_random_clusters_));
        in.read(reinterpret_cast<char *>(&min_size_clusters_), sizeof(min_size_clusters_));
        in.read(reinterpret_cast<char *>(&max_mst_degree_), sizeof(max_mst_degree_));
        adj_memory_.resize(cur_element_count_);
        for (auto& adj: adj_memory_)
        {
          size_t n;
          in.read(reinterpret_cast<char *>(&n), sizeof(n));
          adj.resize(n);
          in.read(reinterpret_cast<char *>(adj.data()), n * sizeof(id_t));
        }
        link_list_locks_.resize(cur_element_count_);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](std::unique_ptr<std::mutex> &lock)
                      { lock = std::make_unique<std::mutex>(); });
        data_memory_.reserve(cur_element_count_);
        for (size_t i = 0; i < cur_element_count_; i++)
        {
          data_memory_.emplace_back(base.data() + i * D_);
        }
      }

      void Save(const std::string& filename) const noexcept
      {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open())
        {
          throw std::runtime_error("Cannot open file for writing");
        }
        out.write(reinterpret_cast<const char *>(&cur_element_count_), sizeof(cur_element_count_));
        out.write(reinterpret_cast<const char *>(&D_), sizeof(D_));
        out.write(reinterpret_cast<const char *>(&num_random_clusters_), sizeof(num_random_clusters_));
        out.write(reinterpret_cast<const char *>(&min_size_clusters_), sizeof(min_size_clusters_));
        out.write(reinterpret_cast<const char *>(&max_mst_degree_), sizeof(max_mst_degree_));
        for (const auto& neighbors: adj_memory_)
        {
          size_t n = neighbors.size();
          const char* buffer = reinterpret_cast<const char *>(neighbors.data());
          out.write(reinterpret_cast<const char *>(&n), sizeof(n));
          out.write(buffer, neighbors.size() * sizeof(id_t));
        }
      }

      size_t GetNumThreads() const noexcept
      {
        return num_threads_;
      }

      void SetNumThreads(size_t num_threads) noexcept
      {
        num_threads_ = num_threads;
      }

      std::vector<std::vector<Edge>> CreateExactMST(
          const std::vector<id_t> &idx_points,
          size_t left, size_t right, size_t max_mst_degree)
      {
        size_t num_points = right - left + 1;
        std::vector<Edge> full_graph;
        std::vector<std::vector<Edge>> mst(num_points);
        full_graph.reserve(num_points * (num_points - 1));

        // pick up all edges into full_graph
        for (size_t i = 0; i < num_points; i++)
        {
          for (size_t j = 0; j < num_points; j++)
          {
            if (i != j)
            {
              full_graph.emplace_back(
                  Edge{i, j, distance(data_memory_[idx_points[left + i]], data_memory_[idx_points[left + j]], D_)});
            }
          }
        }

        // Kruskal algorithm
        std::sort(full_graph.begin(), full_graph.end());
        DisjointSet ds(num_points);
        for (const auto &e : full_graph)
        {
          id_t src = e.src;
          id_t dst = e.dst;
          float weight = e.weight;
          if (ds.Find(src) != ds.Find(dst) && mst[src].size() < max_mst_degree && mst[dst].size() < max_mst_degree)
          {
            mst[src].emplace_back(e);
            mst[dst].emplace_back(Edge{dst, src, weight});
            ds.UnionSet(src, dst);
          }
        }

        return mst;
      }

      void BuildIndex(const std::vector<data_t> &raw_data)
      {
        size_t num_points = raw_data.size() / D_;
        cur_element_count_ = num_points;
        data_memory_.resize(num_points);
        adj_memory_.resize(num_points);
        link_list_locks_.resize(num_points);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](std::unique_ptr<std::mutex> &lock)
                      { lock = std::make_unique<std::mutex>(); });

// initialize graph data
#pragma omp parallel for schedule(dynamic, 16) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          data_memory_[id] = raw_data.data() + id * D_;
          adj_memory_[id].reserve(max_mst_degree_ * num_random_clusters_);
        }
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < num_random_clusters_; i++)
        {
          auto idx_points = std::make_unique<std::vector<id_t>>(num_points);
          for (size_t j = 0; j < num_points; j++)
          {
            idx_points->at(j) = j;
          }
          CreateClusters(*idx_points, 0, num_points - 1, min_size_clusters_, max_mst_degree_);
        }
      }

      void BuildIndex(const std::vector<const data_t *> &raw_data)
      {
        size_t num_points = raw_data.size();
        cur_element_count_ = num_points;
        data_memory_.resize(num_points);
        adj_memory_.resize(num_points);

// initialize graph data
#pragma omp parallel for schedule(dynamic, 16) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
          data_memory_ = raw_data[id];
          adj_memory_[id].reserve(max_mst_degree_ * num_random_clusters_);
        }
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < num_random_clusters_; i++)
        {
          auto idx_points = std::make_unique<std::vector<id_t>>(num_points);
          for (size_t j = 0; j < num_points; j++)
          {
            idx_points->at(j) = j;
          }
          CreateClusters(*idx_points, 0, num_points - 1, min_size_clusters_, max_mst_degree_);
        }
      }

      void CreateClusters(std::vector<id_t> &idx_points, size_t left, size_t right, size_t min_size_clusters, size_t max_mst_degree)
      {
        size_t num_points = right - left + 1;
        if (num_points <= min_size_clusters)
        {
          auto mst = CreateExactMST(idx_points, left, right, max_mst_degree);

          // Add edges to graph
          for (size_t i = 0; i < num_points; i++)
          {
            for (size_t j = 0; j < mst[i].size(); j++)
            {
              std::unique_lock<std::mutex> lock0(*link_list_locks_[idx_points[left + i]]);

              bool is_neighbor = false;
              auto &neigh0 = adj_memory_[idx_points[left + i]];

              for (const auto &nid0 : neigh0)
              {
                if (nid0 == idx_points[left + mst[i][j].dst])
                {
                  is_neighbor = true;
                  break;
                }
              }
              if (!is_neighbor)
              {
                neigh0.emplace_back(idx_points[left + mst[i][j].dst]);
              }
            }
          }
        }
        else
        {
          auto rand_int = [](size_t Min, size_t Max)
          {
            size_t sz = Max - Min + 1;
            return Min + (std::rand() % sz);
          };

          size_t x = rand_int(left, right);
          size_t y = -1;
          do
          {
            y = rand_int(left, right);
          } while (x == y);

          const data_t *vec_idx_left_p_x = data_memory_[idx_points[x]];
          const data_t *vec_idx_left_p_y = data_memory_[idx_points[y]];

          std::vector<id_t> ids_x_set, ids_y_set;
          ids_x_set.reserve(num_points);
          ids_y_set.reserve(num_points);

          for (size_t i = 0; i < num_points; i++)
          {
            const data_t *vec_idx_left_p_i = data_memory_[idx_points[left + i]];

            float dist_x = distance(vec_idx_left_p_x, vec_idx_left_p_i, D_);
            float dist_y = distance(vec_idx_left_p_y, vec_idx_left_p_i, D_);

            if (dist_x < dist_y)
            {
              ids_x_set.emplace_back(idx_points[left + i]);
            }
            else
            {
              ids_y_set.emplace_back(idx_points[left + i]);
            }
          }

          assert(ids_x_set.size() + ids_y_set.size() == num_points);

          // reorder idx_points
          size_t i = 0, j = 0;
          while (i < ids_x_set.size())
          {
            idx_points[left + i] = ids_x_set[i];
            i++;
          }
          while (j < ids_y_set.size())
          {
            idx_points[left + i] = ids_y_set[j];
            j++;
            i++;
          }

          CreateClusters(idx_points, left, left + ids_x_set.size() - 1, min_size_clusters, max_mst_degree);
          CreateClusters(idx_points, left + ids_x_set.size(), right, min_size_clusters, max_mst_degree);
        }
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

        id_t ep = random() % cur_element_count_;

        float dist = distance(data_point, data_memory_[ep], D_);
        comparison++;

        top_candidates.emplace(dist, ep); // max heap
        candidate_set.emplace(-dist, ep); // min heap
        mass_visited[ep] = true;

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
          const auto &neighbors = adj_memory_[curr_node_id];

          for (id_t neighbor_id : neighbors)
          {
            if (!mass_visited[neighbor_id])
            {
              mass_visited[neighbor_id] = true;

              float dd = distance(data_point, data_memory_[neighbor_id], D_);
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

      void PruneNeigh(size_t max_neigh)
      {
#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (id_t id = 0; id < cur_element_count_; id++)
        {
          const data_t *vec_curid = data_memory_[id];

          auto &neigh = adj_memory_[id];

          size_t new_size = std::min(neigh.size(), max_neigh);

          if (new_size == neigh.size())
            continue;

          std::vector<std::pair<float, id_t>> score;
          score.reserve(neigh.size());
          for (const auto &nid : neigh)
          {
            score.emplace_back(distance(data_memory_[nid], vec_curid, D_), nid);
          }

          std::sort(score.begin(), score.end());
          score.resize(new_size);
          score.shrink_to_fit();
          neigh.resize(new_size);
          neigh.shrink_to_fit();

          for (size_t i = 0; i < new_size; i++)
          {
            neigh[i] = score[i].second;
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
        for (id_t id = 0; id < cur_element_count_; id++)
        { // adj list
          sz += adj_memory_[id].size() * sizeof(id_t);
        }
        return sz;
      }
    };

  } // namespace graph

} // namespace index
