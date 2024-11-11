# ANNS Library Usage

This is a very basic library of ANNS sota algorithms. It is written in C++/C++ standard libarary and can be used to build and query an index of points in high-dimensional space. The library provides a variety of algorithms for building and querying the index. The library is designed to be easy to use and to provide relatively high performance.

It's a very useful tool for you to build your own ANNS index alogrithm without too much effort.
There are four graph-based methods and IVFFlat supported now.

## C++ Version

This is an example of how to use the ANNS library to build and query an index.

```c++
DataSetWrapper<data_t> base, query;
  IntervalSetWrapper bcond, qcond;
  GroundTruth gt;
  base.load("data/sift-128-euclidean.train.fvecs");
  query.load("data/sift-128-euclidean.test.fvecs");
  bcond.load("data/sift-128-euclidean.train.uniform-0-1.fvecs");
  qcond.load("data/sift-128-euclidean.test.uniform-0-1.fvecs");
  gt.load("data/sift-128-euclidean.cover.uniform-0-1.ivecs");

  const size_t k = 1;
  utils::Timer timer;

  // build index
  HNSW<data_t, metrics::euclidean> index(32, 128);
  index.set_num_threads(24); 
  timer.start();
  index.build(base, bcond);
  timer.stop();
  cout << "Build time: " << timer.get() << endl;

  // query with different parameters
  ofstream out("hnsw_postfilter.csv");
  for (size_t ef = 1; ef <= 128; ef++)
  {
    timer.reset();
    matrix_id_t knn;
    matrix_di_t dis;
    timer.start();
    index.search(query, qcond, k, ef, 10, Interval::cover, knn, dis);
    timer.stop();
    out << timer.get() << "," << gt.recall(k, knn) << endl;
  }
}
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
