# Benchmark Hnswlib

[run_hnswlib_bench.cpp](./run_hnswlib_bench.cpp) is a multi-thread benchmark program for Hnswlib.
It constructs an HNSW index and performs a set of queries on the index.

The program supports L2 distance and float and uint8_t feature element type.
For example, this program can work on most of the datasets available [here](https://big-ann-benchmarks.com/neurips21.html).

The input files must be in the white-space-separated value (WSV) format with point IDs.
A single line must contain the point ID followed by the feature elements.
To generate input files in this format from the original data, we provide a Python script [convert_big_ann_bench_dataset.py](../convert_big_ann_bench_dataset.py).


## Build

Here is how to build the benchmark program.
This directory is independent of the main SALTATLAS build tree and 
leverages Hnswlib's build script to build the program.

```shell
cd saltfish/tools/hnswlib_bench

git clone git@github.com:nmslib/hnswlib.git
cd hnswlib
git checkout v0.8.0

mkdir bench
cp ../run_hnswlib_bench.cpp ../hnswlib_bench.hpp ./bench

git am ../0001-Build-bench-program.patch

mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release
make run_hnswlib_bench_float run_hnswlib_bench_u8i
```


## Run

`run_hnswlib_bench_float` and `run_hnswlib_bench_u8i` are the executables for float and uint8_t feature element types, respectively.

```shell
./run_hnswlib_bench_float -p ../../../../examples/datasets/point_5-4_id.txt -d 5 -n 20 -m 48 -c 200 -k 4 -e 10:20:30 -q ../../../../examples/datasets/query_5-4.txt -g ../../../../examples/datasets/ground-truth_5-4.txt
```

Available options:
- `-p` Path to an input dataset file or a directory that contains dataset files.
- `-d` Number of dimensions of the input dataset.
- `-n` Number of points in the input dataset.
- `-m` `M` parameter in Hnswlib.
- `-c` `ef_construction` parameter in Hnswlib.
- `-k` Number of nearest neighbors to search.
- `-e` Comma-separated list of `ef` parameters in Hnswlib.
- `-q` Path to a query file.
- `-g` Path to a ground truth file.