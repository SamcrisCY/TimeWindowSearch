#pragma once

#include <vector>
#include "utils.h"
#include "searcher.hpp"
#include "memory.hpp"
#include <bitset>

namespace iRangeGraph
{
    template <typename dist_t>
    class iRangeGraph_Search
    {
    public:
        DataLoader *storage;
        SegmentTree *tree;
        size_t max_elements_{0};
        size_t dim_{0};
        size_t M_out{0};
        size_t ef_construction{0};

        size_t size_data_per_element_{0};
        size_t size_links_per_element_{0};
        size_t data_size_{0};

        size_t size_links_per_layer_{0};
        size_t offsetData_{0};
        bool use_fusion = false;
        char *data_memory_{nullptr};

        //根据向量距离和时间戳的差调整
        double alpha = 1 / 86400;

        std::vector<std::vector<std::vector<std::pair<float, int>>>> edges;
        //edges[i][j][k]表示第j层 ui的第k个邻居编号
        hnswlib::L2Space *space;
        hnswlib::DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_{nullptr};

        size_t metric_distance_computations{0};
        size_t metric_hops{0};

        int prefetch_lines{0};
        
        iRangeGraph_Search(std::vector<std::vector<std::vector<std::pair<float, int>>>> _edges,
                        DataLoader *store, int M)
            : storage(store), edges(_edges)
        {
            max_elements_ = store->data_nb;
            dim_ = store->Dim;

            tree = new SegmentTree(max_elements_);
            tree->BuildTree(tree->root);

            space = new hnswlib::L2Space(dim_);
            fstdistfunc_ = space->get_dist_func();
            dist_func_param_ = space->get_dist_func_param();
            M_out = M;

            data_size_ = (dim_ + 7) / 8 * 8 * sizeof(float);
            size_links_per_layer_ = M_out * sizeof(tableint) + sizeof(linklistsizeint);
            size_links_per_element_ = (size_links_per_layer_ * (tree->max_depth + 1) + 31) / 32 * 32;
            size_data_per_element_ = size_links_per_element_ + data_size_;
            offsetData_ = size_links_per_element_;
            prefetch_lines = data_size_ >> 4;

            //  分配 data_memory_
            data_memory_ = (char *)memory::align_mm<1 << 21>(max_elements_ * size_data_per_element_);
            if (data_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            //  将 edges 写入 data_memory_
            for (int pid = 0; pid < max_elements_; pid++)
            {
                for (int layer = 0; layer <= tree->max_depth; layer++)
                {
                    linklistsizeint *data = get_linklist(pid, layer);

                    // 取出该节点该层的边
                    const auto &layer_edges = edges[pid][layer];
                    int size = static_cast<int>(layer_edges.size());
                    if (size > M_out)
                        throw Exception("edge size exceeds M_out, pid is " + std::to_string(pid) + ", layer is " + std::to_string(layer));

                    // 写入邻居数量
                    *((int *)data) = size;

                    // 写入每个邻居的 id
                    for (int i = 0; i < size; i++)
                    {
                        char *current_neighbor_ = (char *)(data + 1 + i);
                        int neighbor_id = layer_edges[i].second;
                        std::memcpy(current_neighbor_, &neighbor_id, sizeof(tableint));
                    }
                }

                //  写入向量数据
                char *data = getDataByInternalId(pid);
                const std::vector<float> &vector_data = store->data_points[pid];
                std::memcpy(data, vector_data.data(), dim_ * sizeof(float));
            }

        }



        iRangeGraph_Search(std::string edgefilename, DataLoader *store, int M) : storage(store)
        {

            std::ifstream edgefile(edgefilename, std::ios::in | std::ios::binary);
            if (!edgefile.is_open())
                throw Exception("cannot open " + edgefilename);

            // vectorfile.read((char *)&max_elements_, sizeof(int));
            // vectorfile.read((char *)&dim_, sizeof(int));
            max_elements_ = store->data_nb;
            dim_ = store->Dim;


            tree = new SegmentTree(max_elements_);
            tree->BuildTree(tree->root);

            space = new hnswlib::L2Space(dim_);
            fstdistfunc_ = space->get_dist_func();
            dist_func_param_ = space->get_dist_func_param();
            M_out = M;

            data_size_ = (dim_ + 7) / 8 * 8 * sizeof(float);
            size_links_per_layer_ = M_out * sizeof(tableint) + sizeof(linklistsizeint);
            size_links_per_element_ = (size_links_per_layer_ * (tree->max_depth + 1) + 31) / 32 * 32;
            size_data_per_element_ = size_links_per_element_ + data_size_;
            offsetData_ = size_links_per_element_;
            prefetch_lines = data_size_ >> 4;


            data_memory_ = (char *)memory::align_mm<1 << 21>(max_elements_ * size_data_per_element_);
            if (data_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");
            edges.resize(max_elements_);
            for (int pid = 0; pid < max_elements_; pid++) {
                edges[pid].resize(tree->max_depth + 1);  // 每个节点有 max_depth+1 层
            }
            for (int pid = 0; pid < max_elements_; pid++)
            {
                for (int layer = 0; layer <= tree->max_depth; layer++)
                {
                    linklistsizeint *data = get_linklist(pid, layer);
                    edgefile.read((char *)data, sizeof(tableint));
                    int size = getListCount(data);
                    if (size > M_out)
                        throw Exception("real linklist size is bigger than defined M_out");
                    std::vector<std::pair<float, int>> layer_edges;
                    for (int i = 0; i < size; i++)
                    {
                        char *current_neighbor_ = (char *)(data + 1 + i);
                        edgefile.read(current_neighbor_, sizeof(tableint));
                        int neighbor_id = *((int *)current_neighbor_);

                        //new
                        float distance = 0.0f; 

                        // 将边信息添加到当前层的邻接边列表中
                        layer_edges.push_back({distance, neighbor_id});
                    }
                    //new
                    edges[pid][layer] = layer_edges;
                }

                char *data = getDataByInternalId(pid);
                // vectorfile.read(data, data_size_);
                
                // vectorfile.read(data, dim_ * sizeof(float));
                //todo
                std::vector<float> vector_data = store->data_points[pid];
                std::memcpy(data, vector_data.data(), dim_ * sizeof(float));
            }

            edgefile.close();
            // vectorfile.close();
        }

        iRangeGraph_Search(std::string vectorfilename, std::string edgefilename, DataLoader *store, int M) : storage(store)
        {
            std::ifstream vectorfile(vectorfilename, std::ios::in | std::ios::binary);
            if (!vectorfile.is_open())
                throw Exception("cannot open " + vectorfilename);
            std::ifstream edgefile(edgefilename, std::ios::in | std::ios::binary);
            if (!edgefile.is_open())
                throw Exception("cannot open " + edgefilename);

            vectorfile.read((char *)&max_elements_, sizeof(int));
            vectorfile.read((char *)&dim_, sizeof(int));

            tree = new SegmentTree(max_elements_);
            tree->BuildTree(tree->root);

            space = new hnswlib::L2Space(dim_);
            fstdistfunc_ = space->get_dist_func();
            dist_func_param_ = space->get_dist_func_param();
            M_out = M;

            data_size_ = (dim_ + 7) / 8 * 8 * sizeof(float);
            size_links_per_layer_ = M_out * sizeof(tableint) + sizeof(linklistsizeint);
            size_links_per_element_ = (size_links_per_layer_ * (tree->max_depth + 1) + 31) / 32 * 32;
            size_data_per_element_ = size_links_per_element_ + data_size_;
            offsetData_ = size_links_per_element_;
            prefetch_lines = data_size_ >> 4;

            data_memory_ = (char *)memory::align_mm<1 << 21>(max_elements_ * size_data_per_element_);
            if (data_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            for (int pid = 0; pid < max_elements_; pid++)
            {
                for (int layer = 0; layer <= tree->max_depth; layer++)
                {
                    linklistsizeint *data = get_linklist(pid, layer);
                    edgefile.read((char *)data, sizeof(tableint));
                    int size = getListCount(data);
                    if (size > M_out)
                        throw Exception("real linklist size is bigger than defined M_out");
                    for (int i = 0; i < size; i++)
                    {
                        char *current_neighbor_ = (char *)(data + 1 + i);
                        edgefile.read(current_neighbor_, sizeof(tableint));
                        
                    }
                }

                char *data = getDataByInternalId(pid);
                // vectorfile.read(data, data_size_);
                vectorfile.read(data, dim_ * sizeof(float));
            }

            edgefile.close();
            vectorfile.close();
        }

        ~iRangeGraph_Search()
        {
            free(data_memory_);
            data_memory_ = nullptr;
        }

        inline int mapLayer(int depth) const {
            return tree->max_depth - depth;
        }


        inline char *getDataByInternalId(tableint internal_id) const
        {
            return (data_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        linklistsizeint *get_linklist(tableint internal_id, int layer) const
        {
            return (linklistsizeint *)(data_memory_ + internal_id * size_data_per_element_ + layer * size_links_per_layer_);
        }

        int getListCount(linklistsizeint *ptr) const
        {
            return *((int *)ptr);
        }

        int GetOverLap(int l, int r, int ql, int qr)
        {
            int L = std::max(l, ql);
            int R = std::min(r, qr);
            return R - L + 1;
        }

        std::vector<tableint> SelectEdge(int pid, int ql, int qr, int edge_limit, searcher::Bitset<uint64_t> &visited_set)
        {
            TreeNode *cur_node = nullptr, *nxt_node = tree->root;
            std::vector<tableint> selected_edges;
            selected_edges.reserve(edge_limit);
            do
            {
                cur_node = nxt_node;
                bool contain = false;
                do
                {
                    contain = false;
                    if (cur_node->childs.size() == 0)
                        nxt_node = nullptr;
                    else
                    {
                        for (int i = 0; i < cur_node->childs.size(); ++i)
                        {
                            if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid)
                            {
                                nxt_node = cur_node->childs[i];
                                break;
                            }
                        }
                        if (GetOverLap(cur_node->lbound, cur_node->rbound, ql, qr) == GetOverLap(nxt_node->lbound, nxt_node->rbound, ql, qr))
                        {
                            cur_node = nxt_node;
                            contain = true;
                        }
                    }
                } while (contain);

                // int *data = (int *)get_linklist(pid, cur_node->depth);
                // size_t size = getListCount((linklistsizeint *)data);

                // for (size_t j = 1; j <= size; ++j)
                // {
                //     int neighborId = *(data + j);
                //     if (neighborId < ql || neighborId > qr)
                //         continue;
                //     // if (visitedpool[neighborId] == visited_tag)
                //     //     continue;
                //     if (visited_set.get(neighborId))
                //         continue;
                //     selected_edges.emplace_back(neighborId);
                //     if (selected_edges.size() == edge_limit)
                //         return selected_edges;
                // }
                if(pid >= edges.size()){
                    std::cout << "pid: " << pid  << ", edges size:" << edges.size() << std::endl;
                    throw Exception("vector id exceeds");
                }else if(edges[pid].size() <= cur_node->depth){
                    throw Exception("depth exceeds");
                }
                const auto &layer_edges = edges[pid][mapLayer(cur_node->depth)];
                for (const auto &neighbor : layer_edges)
                {
                    int neighborId = neighbor.second;
                    if (neighborId < ql || neighborId > qr)
                        continue;
                    if (visited_set.get(neighborId))
                        continue;

                    selected_edges.emplace_back(neighborId);
                    if (selected_edges.size() == edge_limit)
                        return selected_edges;
                }

            } while (cur_node->lbound < ql || cur_node->rbound > qr);

            return selected_edges;
        }

        std::priority_queue<PFI> TopDown_nodeentries_search(std::vector<TreeNode *> &filterednodes, const void *query_data, int ef, int query_k, int QL, int QR, int edge_limit, double ten = -1)
        {
            // To fix the starting points for different 'ef' parameter, set seed to a fixed number, e.g., seed =0
            // unsigned seed = 0;
            
            //the random entry point make results not deterministic
            
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);
            
            std::priority_queue<PFI, std::vector<PFI>, std::greater<PFI>> candidate_set; // BFS队列
            std::priority_queue<PFI> top_candidates; //结果队列
            searcher::Bitset<uint64_t> visited_set(max_elements_);

            for (auto u : filterednodes)
            {
                std::uniform_int_distribution<int> u_start(u->lbound, u->rbound);
                int pid = u_start(e);
                visited_set.set(pid);
                char *ep_data = getDataByInternalId(pid);
                float dis = fstdistfunc_(query_data, ep_data, dist_func_param_);
                
                // if(use_fusion && storage->t_hash[pid] >= ten){
                //     dis += (storage->t_hash[pid] - ten) * k;
                // }

                top_candidates.emplace(dis, pid);
                candidate_set.emplace(dis, pid);
            }

            float lowerBound = top_candidates.top().first;

            while (!candidate_set.empty())
            {
                auto current_point_pair = candidate_set.top();
                ++metric_hops;
                if (current_point_pair.first > lowerBound)
                {
                    break;
                }
                candidate_set.pop();
                int current_pid = current_point_pair.second;
                auto selected_edges = SelectEdge(current_pid, QL, QR, edge_limit, visited_set);
                int num_edges = selected_edges.size();
                for (int i = 0; i < std::min(num_edges, 3); ++i)
                {
                    memory::mem_prefetch_L1(getDataByInternalId(selected_edges[i]), this->prefetch_lines);
                }
                for (int i = 0; i < num_edges; ++i)
                {
                    int neighbor_id = selected_edges[i];

                    if (visited_set.get(neighbor_id))
                        continue;
                    visited_set.set(neighbor_id);
                    char *neighbor_data = getDataByInternalId(neighbor_id);
                    float dis = fstdistfunc_(query_data, neighbor_data, dist_func_param_);
                    // if(use_fusion && storage->fusion_hash[neighbor_id] >= ten){
                    //     dis += (storage->fusion_hash[neighbor_id] - ten) * alpha;
                    // }
                    ++metric_distance_computations;

                    if (top_candidates.size() < ef)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        if(!use_fusion || storage->fusion_hash[neighbor_id] < ten)
                            top_candidates.emplace(dis, neighbor_id);
                        lowerBound = top_candidates.top().first;
                    }
                    else if (dis < lowerBound)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        if(!use_fusion || storage->fusion_hash[neighbor_id] < ten)
                        {
                            top_candidates.emplace(dis, neighbor_id);
                            top_candidates.pop();
                            lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }
            // while (top_candidates.size() > query_k && 
            // !storage->vid_hash[top_candidates.top().second].second)
            //     top_candidates.pop();
            
            return top_candidates;
        }

        std::vector<PFI> search(int ef, int edge_limit, int l, int r, std::vector<float> query_vector, bool _use_fusion = false, double ten = -1){
                use_fusion = _use_fusion;
                std::vector<TreeNode *> filterednodes = tree->range_filter(tree->root, l, r);
                std::priority_queue<PFI> res = TopDown_nodeentries_search(filterednodes, query_vector.data(), ef, ef, l, r, edge_limit, ten);
                // std::cout << "Get result successfully" << std::endl;
                std::vector<PFI> globalres;
                while(!res.empty()){
                    if((!use_fusion && !storage->vid_hash[res.top().second].second) || (use_fusion && storage->fusion_hash[res.top().second] <= ten)){
                        globalres.push_back({res.top().first, storage->vid_hash[res.top().second].first});
                    }
                    // if(storage->vid_hash[res.top().second].first == 11935){
                    //     std::cout << storage->vid_hash[res.top().second].first << ": " << res.top().first << std::endl;
                    // }
                    res.pop();
                }
                use_fusion = false;
                return globalres;
        }

        void search(std::vector<int> &SearchEF, std::string saveprefix, int edge_limit)
        {
            for (auto range : storage->query_range)
            {
                int suffix = range.first;
                std::vector<std::vector<int>> &gt = storage->groundtruth[suffix];
                std::string savepath = saveprefix + std::to_string(suffix) + ".csv";
                // CheckPath(savepath);
                std::ofstream outfile(savepath);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);

                std::vector<int> HOP;
                std::vector<int> DCO;
                std::vector<float> QPS;
                std::vector<float> RECALL;

                std::cout << "suffix = " << suffix << std::endl;
                for (auto ef : SearchEF)
                {
                    int tp = 0;
                    float searchtime = 0;

                    metric_hops = 0;
                    metric_distance_computations = 0;

                    for (int i = 0; i < storage->query_nb; i++)
                    {
                        auto rp = range.second[i];
                        int ql = rp.first, qr = rp.second;
                        timeval t1, t2;
                        gettimeofday(&t1, NULL);
                        std::vector<TreeNode *> filterednodes = tree->range_filter(tree->root, ql, qr);
                        std::priority_queue<PFI> res = TopDown_nodeentries_search(filterednodes, storage->query_points[i].data(), ef, storage->query_K, ql, qr, edge_limit);
                        gettimeofday(&t2, NULL);
                        auto duration = GetTime(t1, t2);
                        searchtime += duration;
                        std::map<int, int> record;
                        while (res.size())
                        {
                            auto x = res.top().second;
                            res.pop();
                            if (record.count(x))
                                throw Exception("repetitive search results");
                            record[x] = 1;
                            if (std::find(gt[i].begin(), gt[i].end(), x) != gt[i].end())
                                tp++;
                        }
                    }

                    float recall = 1.0 * tp / storage->query_nb / storage->query_K;
                    float qps = storage->query_nb / searchtime;
                    float dco = metric_distance_computations * 1.0 / storage->query_nb;
                    float hop = metric_hops * 1.0 / storage->query_nb;

                    HOP.emplace_back(hop);
                    DCO.emplace_back(dco);
                    QPS.emplace_back(qps);
                    RECALL.emplace_back(recall);
                }

                for (int i = 0; i < RECALL.size(); i++)
                {
                    outfile << SearchEF[i] << "," << RECALL[i] << "," << QPS[i] << "," << DCO[i] << "," << HOP[i] << std::endl;
                }
                outfile.close();
            }
        }
    };
}