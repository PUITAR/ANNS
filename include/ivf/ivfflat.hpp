#pragma once

#include <base.hpp>
#include <anns.hpp>
#include <vector>
#include <queue>
#include <mutex>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <distance.hpp>
#include <random>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

namespace anns
{

  namespace ivf
  {

    template <typename data_t, float (*distance)(const data_t *, const data_t *, size_t)>
    class IVFFlat : public Base<data_t>
    {
    protected:
      size_t kc_{0};
      size_t it_{0};
      std::vector<data_t> centroids_;
      std::vector<size_t> offset_;
      std::vector<id_t> reo_;

    public:
      __USE_BASE__

      IVFFlat(size_t kc, size_t it) : kc_(kc), it_(it) {}

      /// @todo build from file
      IVFFlat(const DataSet<data_t> &base, const std::string &filename) noexcept
      {
        base_ = base;
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open())
        {
          throw std::runtime_error("IVFFlat: failed to open file for reading");
        }
        in.read(reinterpret_cast<char *>(&kc_), sizeof(size_t));
        in.read(reinterpret_cast<char *>(&it_), sizeof(size_t));
        centroids_.resize(base_.dim * kc_);
        offset_.resize(kc_);
        reo_.resize(base_.num_);
        for (size_t i = 0; i < base_.dim * kc_; i++)
        {
          in.read(reinterpret_cast<char *>(&centroids_[i]), sizeof(data_t));
        }
        for (size_t i = 0; i < kc_; i++)
        {
          in.read(reinterpret_cast<char *>(&offset_[i]), sizeof(size_t));
        }
        for (size_t i = 0; i < base_.num_; i++)
        {
          in.read(reinterpret_cast<char *>(&reo_[i]), sizeof(id_t));
        }
      }

      void save(const std::string &filename) const noexcept override
      {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open())
        {
          throw std::runtime_error("IVFFlat: failed to open file for writing");
        }
        out.write(reinterpret_cast<const char *>(&kc_), sizeof(kc_));
        out.write(reinterpret_cast<const char *>(&it_), sizeof(it_));
        for (size_t i = 0; i < centroids_.size(); i++)
        {
          out.write(reinterpret_cast<const char *>(&centroids_[i]), sizeof(data_t));
        }
        for (size_t i = 0; i < offset_.size(); i++)
        {
          out.write(reinterpret_cast<const char *>(&offset_[i]), sizeof(size_t));
        }
        for (size_t i = 0; i < reo_.size(); i++)
        {
          out.write(reinterpret_cast<const char *>(&reo_[i]), sizeof(id_t));
        }
      }

      void build(const DataSet<data_t> &base) noexcept override
      {
        base_ = base;
        assert(
            kc_ <= base_.num_ &&
            "IVFFlat: kc must be less than or equal to the number of data points");
        centroids_.resize(kc_ * base_.dim_);
        offset_.resize(kc_);
        reo_.resize(base_.num_);
        // initialize centroids
        data_t mi = std::numeric_limits<data_t>::max(),
               ma = std::numeric_limits<data_t>::min();
        for (size_t i = 0; i < base_.num_; i++)
        {
          for (size_t j = 0; j < base_.dim_; j++)
          {
            mi = std::min(mi, base_[i][j]);
            ma = std::max(ma, base_[i][j]);
          }
        }
        // std::cout << 112 << std::endl;
#pragma omp parallel for num_threads(num_threads_) schedule(static)
        for (size_t i = 0; i < kc_; i++)
        {
          for (size_t j = 0; j < base_.dim_; j++)
          {
            centroids_[i * base_.dim_ + j] = get_random_number(mi, ma);
          }
        }
        // std::cout << 121 << std::endl;
        // training
        std::vector<std::mutex> mutexes(kc_);
        std::vector<id_t> cids(base_.num_);
        for (size_t tt = 0; tt < it_; tt++)
        {
          // find nearest centroid
          std::fill(offset_.begin(), offset_.end(), 0);
#pragma omp parallel for num_threads(num_threads_) schedule(static)
          for (id_t id = 0; id < base_.num_; id++)
          {
            cids[id] = find_nearest_centroid(base_[id], 1).top().second;
            std::unique_lock<std::mutex> lock(mutexes[cids[id]]);
            offset_[cids[id]]++;
          }
          // std::cout << 136 << std::endl;
          // update centroids
          std::fill(centroids_.begin(), centroids_.end(), 0);
#pragma omp parallel for num_threads(num_threads_) schedule(static)
          for (id_t id = 0; id < base_.num_; id++)
          {
            std::unique_lock<std::mutex> lock(mutexes[cids[id]]);
            for (size_t j = 0; j < base_.dim_; j++)
            {
              centroids_[cids[id] * base_.dim_ + j] += base_[id][j];
            }
          }
          // std::cout << 148 << std::endl;
#pragma omp parallel for num_threads(num_threads_) schedule(static)
          for (size_t i = 0; i < kc_; i++)
          {
            for (size_t j = 0; j < base_.dim_; j++)
            {
              centroids_[i * base_.dim_ + j] /= offset_[i];
            }
          }
          // std::cout << 157 << std::endl;
        }
        // std::cout << 158 << std::endl;
        // update offsets
        for (size_t i = 1; i < kc_; i++)
        {
          offset_[i] += offset_[i - 1];
        }
        // std::cout << 165 << std::endl;
        // reorder vectors
        for (id_t id = 0; id < base_.num_; id++)
        {
          reo_[id] = id;
        }
        std::sort(reo_.begin(), reo_.end(), [&](id_t a, id_t b)
                  { return cids[a] < cids[b]; });
        base_.hash_ = [this](id_t id)
        { return this->reo_[id]; };
        // std::cout << 175 << std::endl;
      }

      size_t index_size() const noexcept override
      {
        return centroids_.size() * sizeof(data_t) + offset_.size() * sizeof(size_t) + reo_.size() * sizeof(id_t);
      }

      void search(const DataSet<data_t> &query, size_t k, size_t nprobe, matrix_id_t &knn, matrix_di_t &dis) override
      {
        assert(base_.dim_ == query.dim_);
        knn.clear(), knn.resize(query.num_);
        dis.clear(), dis.resize(query.num_);

#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads_)
        for (size_t i = 0; i < query.num_; i++)
        {
          const data_t *q = query[i];
          auto &knn_i = knn[i];
          auto &dis_i = dis[i];
          auto nnc = find_nearest_centroid(q, nprobe);
          std::priority_queue<std::pair<float, id_t>> candidates;
          while (nnc.size())
          {
            auto [sid, eid] = cluster_at(nnc.top().second);
            nnc.pop();
            for (id_t id = sid; id < eid; id++)
            {
              candidates.emplace(distance(q, base_[id], base_.dim_), id);
              if (candidates.size() > k)
                candidates.pop();
            }
          }
          knn_i.reserve(candidates.size());
          dis_i.reserve(candidates.size());
          while (candidates.size())
          {
            auto [d, id] = candidates.top();
            candidates.pop();
            knn_i.emplace_back(reo_[id]);
            dis_i.emplace_back(d);
          }
        }
      }

    protected:
      inline std::priority_queue<std::pair<float, id_t>>
      find_nearest_centroid(const data_t *x, size_t k) const noexcept
      {
        std::priority_queue<std::pair<float, id_t>> pq;
        for (id_t i = 0; i < kc_; i++)
        {
          pq.emplace(distance(x, centroids_.data() + i * base_.dim_, base_.dim_), i);
          if (pq.size() > k)
            pq.pop();
        }
        return pq;
      }

      inline std::pair<id_t, id_t> cluster_at(id_t id) const noexcept
      {
        return {id ? offset_[id - 1] : 0, offset_[id]};
      }

      inline auto get_random_number(data_t mi, data_t ma) -> data_t
      {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<data_t> dis(mi, ma);
        return dis(gen);
      };
    };

  }

}
