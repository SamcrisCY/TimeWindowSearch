# BiGraph: A Dual-Index Structure for Temporal-Constrained Vector Similarity Search
## Abstract
In practice, unstructured data in domains where timeliness matters, such as in legal documents and social media posts, is often represented as embedding vectors paired with validity time intervals. This motivates the **T**emporal-**C**onstrained **V**ector **S**imilarity Search (TCVSS) problem: retrieving the nearest neighbors of a query vector from objects whose validity intervals overlap with a given time window. TCVSS poses two core challenges: (1) efficiently managing dynamic updates and (2) performing low-latency similarity search under temporal constraints. Existing methods are either inflexible for dynamic updates or exhibit unsatisfactory search performance.
To overcome these limitations, we propose BiGraph, a novel dual-index framework. BiGraph organizes temporal vectors by validity state using two dedicated graph indexes: a dynamic index for currently valid vectors and an append-only index for expired vectors. This decoupled design supports efficient data maintenance and decomposes TCVSS queries into simpler sub-queries, enabling low-latency search. Extensive experiments on real-world datasets demonstrate that BiGraph outperforms state-of-the-art methods, achieving over 7$\times$ higher query throughput while maintaining equivalent or superior recall.
## Datasets
Download and preprocess the datasets. The tested datasets are available at
https://pan.baidu.com/s/5Wxi9jjc5vidAGcIvQeJOPQ
## Compile and run our algorithms
### build and make
```bash
mkdir build && cd build && cmake .. && make
```
### run code
```bash
./tests/test_search --vector_path [path of vectors] --time_path [path of temporal attributes] --M [out-degree of graph-based index] --ef_construction [parameter of graph-based index] --Dim [dimensionality of vectors] --log_path [path to store update results] -- res_path[path to store search results] --query_path [path of queries] --percent [run percent% update operations]

