# Approximation-k-median-clustering

This repository provides a **Range Counting Oracle** data structure and an **approximation k-median clustering** algorithm. Given a point set $P$ of $n$ points, the algorithm selects $k$ centers $C$ to minimize the total distance from each point to its nearest center:

$$
\mathrm{cost}(P, C) \;=\; \sum_{p \in P} \min_{c \in C} \lVert p - c \rVert .
$$

Note that we implemented these algorithms for experiments supporting the main research and development of our project, and all implementations in this repository are written in **C++ (C++17)**.
