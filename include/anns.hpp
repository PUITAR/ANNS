#pragma once

#include <numeric>
#include <vector>
#include <utility>
#include <functional>

#include <utils/binary.hpp>

namespace anns
{
  using knn_t = std::vector<int>;
  using dis_t = std::vector<float>;
  using res_t = std::pair<dis_t, knn_t>;

#define MAGIC_ID 0x3f3f3f3f
#define MAGIC_DIST std::numeric_limits<float>::max()
#define MAGIC_DIMENSION 1024

#define EPSILON std::numeric_limits<float>::denorm_min()

  inline int DEFAULT_HASH(int id)
  {
    return id;
  }

  /// @brief  {base pointer, num, dimension}
  /// @tparam data_t
  template <typename data_t>
  struct DataSet
  {
    const data_t *data_{nullptr};
    size_t num_{0};
    size_t dim_{0};
    std::function<int(int)> hash_{DEFAULT_HASH};

    inline const data_t *access(int id) const
    {
      return data_ + hash_(id) * dim_;
    }

    inline const data_t *operator[](int id) const
    {
      return data_ + hash_(id) * dim_;
    }
  };

  template <typename data_t>
  struct DataSetWrapper : public DataSet<data_t>
  {
    using DataSet<data_t>::data_;
    using DataSet<data_t>::num_;
    using DataSet<data_t>::dim_;
    using DataSet<data_t>::hash_;
    std::vector<data_t> base_;

    DataSetWrapper(const std::string &fname)
    {
      load(fname);
    }

    void load(const std::string &filename, bool bin = false) noexcept
    {
      if (bin)
        std::tie(num_, dim_) = utils::load_from_file_bin(base_, filename);
      else
        std::tie(num_, dim_) = utils::load_from_file(base_, filename);
      data_ = base_.data();
    }
  };

}