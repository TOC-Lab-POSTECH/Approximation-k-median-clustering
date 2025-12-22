# Approximation-k-median-clustering

This repository provides a Range Counting Oracle data structure and an approximation k-median clustering algorithm. Given a point set $P$ of $n$ points, the algorithm selects $k$ centers $C$ to minimize the total distance from each point to its nearest center. 

Note that we implemented these algorithms for experiments supporting the main research and development of our project, and all implementations in this repository are written in C++ (C++17).

# Requirements
- [boost](https://www.boost.org/)
- [CGAL](https://www.cgal.org/)

Note that CGAL has the dependency on the boost library since it uses the boost graph library. Thus, we recommend to download both libraries together following the package manager of CGAL.

# Data structure & Algorithm
## Range Counting Oracle Data structure
The Range Counting Oracle (RCO) is a data structure, such as range trees, for accessing a large 2D point set without scanning all points explicitly. Instead of reading individual points, the oracle answers queries of the form: "How many points lie inside an axis-aligned rectangle $[x_1, x_2] \times [y_1, y_2]$" This model is useful when direct access to the dataset is expensive or restricted, and we want algorithms whose complexity is measured by the number of oracle queries (rather than by runtime on the full input).
In this project, the RCO is used as the main primitive to support sublinear geometric algorithms by enabling hierarchical space decomposition basend only on rectangle counting queries.

## Approximation k-median clustering Algorithm
The k-median clustering aims to select $k$ centers minimizing the sum of Euclidean distances from each point to its nearest center: $\mathrm{cost}(P, C) = \sum_{p \in P} \min_{c \in C} \lVert p - c \rVert.$ Our goal is to compute an approximate k-median solution in the Range Counting Oracle model, using only a sublinear number of range-counting queries.
The key idea is to avoid running clustering on the full set $P$, and instead construct a small weighted representative set $R_j$ that summarizes the distribution of $P$. This representative set is built via a hierarchical grid/quadtree refinement guided by range-counting queries.
After building $R_j$, we run a standard clustering routine on the much smaller weighted set. The resulting centers are then evaluated on the original dataset. Our implementation is based on the work of [[Monemizadeh 23]](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.APPROX/RANDOM.2023.6) 

# How to run(On Mac)
We use [brew](https://brew.sh/) to install the requirements.
1. install(terminal)
    ```
   brew install cmake
   brew install cgal
   ```
   
2. (If you did not clone this repository, then this step may be required.) Configure CMakeLists.txt(Refer to CMakeLists.txt at the root folder)

4. create and move to build folder(on Project folder)
   ```
   mkdir build
   cd build
   ```
5. run cmake
   ```
   cmake ..
   ```
6. run make to create the execution
   ```
   make
   ```
   
7. run execution.(named Project1)
   ```
   ./Project1
   ```
