# ANNS Library Usage

This is a very basic library of ANNS sota algorithms. It is written in C++/C++ standard libarary and can be used to build and query an index of points in high-dimensional space. The library provides a variety of algorithms for building and querying the index. The library is designed to be easy to use and to provide relatively high performance.

It's a very useful tool for you to build your own ANNS index alogrithm without too much effort.
There are four graph-based methods and IVFFlat supported now.

## C++ Version

This is an example of how to use the ANNS library to build and query an index.

```c++
#include <graph/hnsw.hpp>
#include <utils/binary.hpp>
#include <utils/resize.hpp>
#include <utils/recall.hpp>
#include <utils/timer.hpp>
#include <iostream>

using namespace std;
using namespace anns;

// const std::string bp = "/var/lib/docker/anns/dataset/sift1m/base.fvecs";
const std::string bp = "../data/sift-128-euclidean.train.fvecs";
const std::string qp = "../data/sift-128-euclidean.test.fvecs";
const std::string gp = "../data/sift-128-euclidean.gt.ivecs";

const size_t k = 1;

int main(int argc, char **argv)
{
  std::vector<float> base, query;
  std::vector<id_t> gt;
  auto [nb, d] = utils::load_from_file(base, bp);
  auto [nq, _] = utils::load_from_file(query, qp);
  auto [ng, t] = utils::load_from_file(gt, gp);
  auto nestq = utils::nest(std::move(query), nq, d);
  std::cout << nb << "x" << d << std::endl;
  std::cout << nq << "x" << d << std::endl;
  std::cout << ng << "x" << t << std::endl;

  { /* Build an user-defined index, then save it into the file with suffix '.idx'. */
    auto index = std::make_unique<anns::graph::HNSW<float, metrics::L2>> (d, 16, 500);
    utils::Timer timer;
    timer.start();
    index->set_num_threads(24);
    index->build(base);
    timer.stop();
    std::cout << "build time: " << timer.get() << " s" << std::endl;
    index->save("hnsw_sift1m.idx");
    std::cout << "index size: " << index->index_size() << " bytes" << std::endl;
  }

  { /* Read the index from the index file build so far, then process the k-ANNS task */
    auto index = std::make_unique<anns::graph::HNSW<float, metrics::L2>> (base, "hnsw_sift1m.idx");
    index->set_num_threads(24);
    std::vector<std::vector<id_t>> knn;
    std::vector<std::vector<float>> dis;
    index->search(nestq, k, 128, knn, dis);
    auto recall = utils::recall(k, t, gt, knn);
    std::cout << "recall: " << recall << std::endl;
  }

  return 0;
}
```

## Python Version

To use the ANNS library simply, we wrap the CPP APIs with Python API, which you can use as follow. [Example for ANNS Python API](test_anns.ipynb)

```python
# Still on going...
```

## Dataset 

All datasets used in this library are in the **vecs**-format. If your interested dataset is in the ann-benchmark, we here provid a framework base on the benchmark, to download the datasets and convert them to **vecs**-format. You can find the code in [here](https://github.com/PUITAR/ANNS-DATA.git).

## Delta-Development

If you are interested in the delta-development of the ANN algorithm, you can inhert from a index-based class and implement your own algorithm to keep the original API. For example

```c++

template <typename data_t, typename label_t, float (*distance)(const data_t *, const data_t *, size_t)>
  class PostFilterHCNNG : public HCNNG<data_t, distance>
/* ... */
```
