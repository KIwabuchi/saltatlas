# Benchmark Hnswlib

## Build

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

```shell
./run_hnswlib_bench_float -p ../../../../examples/datasets/point_5-4_id.txt -d 5 -n 20 -m 48 -c 200 -k 4 -e 10:20:30 -q ../../../../examples/datasets/query_5-4.txt -g ../../../../examples/datasets/ground-truth_5-4.txt
```