#pragma once

#include <numeric>
#include <vector>
#include <utility>
#include <functional>

namespace anns
{

#define MAGIC_ID std::numeric_limits<id_t>::max()
#define MAGIC_DIST std::numeric_limits<float>::max()
#define MAGIC_DIMENSION 1024

#define EPSILON std::numeric_limits<float>::denorm_min()

  /// @brief identifiers matrix type (2D)
  using matrix_id_t = std::vector<std::vector<id_t>>;
  /// @brief distances matrix type (2D)
  using matrix_di_t = std::vector<std::vector<float>>;

  inline id_t DEFAULT_HASH(id_t id)
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
    std::function<id_t(id_t)> hash_{DEFAULT_HASH};

    inline const data_t *access(id_t id) const
    {
      return data_ + hash_(id) * dim_;
    }

    inline const data_t *operator[](id_t id) const
    {
      return data_ + hash_(id) * dim_;
    }
  };

}