#pragma once

#include <thread>
#include <map>
#include "utils.h"
#include "searcher.hpp"
#include <bitset>
#include "./utils/DataType.hpp"
#include "./utils/MetricType.hpp"
#include "./utils/BenchLogger.hpp"
#include "./utils/File_IO.h"


namespace iRangeGraph
{
    typedef std::pair<float, int> PFI;
    template <typename dist_t>
    class iRangeGraph_Build
    {
    public:
        size_t max_threads = 1;
        SegmentTree *tree;
        DataLoader *storage;
        std::vector<std::vector<std::vector<std::pair<float, int>>>> edges;
        std::vector<std::vector<PFI>> reverse_edges;
        size_t M;
        size_t ef_construction;

        hnswlib::L2Space *space;
        hnswlib::DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_{nullptr};
        std::vector<size_t> visitedpool;
        size_t visited_tag{0};
        std::mutex tag_mutex;

        int max_depth;

        std::queue<int> threadidpool;

        iRangeGraph_Build(DataLoader *store, int M_out = 32, int ef_c = 400) : storage(store), M(M_out), ef_construction(ef_c)
        {
            space = new hnswlib::L2Space(storage->Dim);
            fstdistfunc_ = space->get_dist_func();
            dist_func_param_ = space->get_dist_func_param();
            tree = new SegmentTree(storage->data_nb);
            tree->BuildTree(tree->root);
            max_depth = tree->max_depth;
            edges.resize(storage->data_nb);
            for (int i = 0; i < storage->data_nb; i++)
            {
                edges[i].resize(tree->max_depth + 1);
            }
            visitedpool.resize(storage->data_nb);
        }
        iRangeGraph_Build(DataLoader *store, int M_out, int ef_c,
        std::vector<std::vector<std::vector<std::pair<float, int>>>> _edges)
         : storage(store), M(M_out), ef_construction(ef_c), edges(_edges)
        {
            space = new hnswlib::L2Space(storage->Dim);
            fstdistfunc_ = space->get_dist_func();
            dist_func_param_ =  space->get_dist_func_param();
            tree = new SegmentTree(storage->data_nb);
            tree->BuildTree(tree->root);
            max_depth = tree->max_depth;
            visitedpool.resize(storage->data_nb);
        }
        // iRangeGraph_Build(DataLoader *store, std::string index_path, int M_out = 32, int ef_c = 400) : storage(store), M(M_out), ef_construction(ef_c)
        // {
        //     space = new hnswlib::L2Space(storage->Dim);
        //     fstdistfunc_ = space->get_dist_func();
        //     dist_func_param_ = space->get_dist_func_param();
        //     tree = new SegmentTree(storage->data_nb);
        //     tree->BuildTree(tree->root);
        //     edges.resize(storage->data_nb);
        //     for (int i = 0; i < storage->data_nb; i++)
        //     {
        //         edges[i].resize(tree->max_depth + 1);
        //     }
        //     visitedpool.resize(storage->data_nb);
        // }

        float dis_compute(std::vector<float> &v1, std::vector<float> &v2)
        {
            return fstdistfunc_(v1.data(), v2.data(), dist_func_param_);
        }

        float dis_compute(VectorDataType &v1, VectorDataType &v2){
            return fstdistfunc_(v1.data.data(), v2.data.data(), dist_func_param_);
        }

        int mapLayer(int depth) const {
            return max_depth - depth;
            // return depth;
        }
        void copyfirstchild(TreeNode *u)
        {
            TreeNode *firstchild = u->childs[0];
            for (int id = firstchild->lbound; id <= firstchild->rbound; id++)
            {
                int lower_layer = mapLayer(firstchild->depth);
                int higher_layer = mapLayer(u->depth);
                edges[id][higher_layer] = edges[id][lower_layer];
            }
        }

        std::priority_queue<PFI> search_on_incomplete_graph(TreeNode *u, std::vector<float> &query_point, int ef, int query_k, std::vector<int> enterpoints)
        {
            size_t local_tag;
            {
                std::lock_guard<std::mutex> lock(tag_mutex);
                visited_tag++;
                local_tag = visited_tag;
            }

            std::priority_queue<PFI, std::vector<PFI>, std::greater<PFI>> pool;
            std::priority_queue<PFI> candidates;

            for (auto pid : enterpoints)
            {
            
                float dis = dis_compute(query_point, storage->data_points[pid]);
                visitedpool[pid] = local_tag;
                pool.emplace(dis, pid);
                candidates.emplace(dis, pid);
            }

            float lowerBound = candidates.top().first;

            int layer = mapLayer(u->depth);

            while (!pool.empty())
            {
                auto current_pair = pool.top();
                if (current_pair.first > lowerBound)
                    break;
                pool.pop();
                int current_pointId = current_pair.second;
                size_t size = edges[current_pointId][layer].size();

                for (int i = 0; i < size; i++)
                {
                    int neighborId = edges[current_pointId][layer][i].second;
                    if (visitedpool[neighborId] == local_tag)
                        continue;
                    visitedpool[neighborId] = local_tag;
                    float dis = dis_compute(query_point, storage->data_points[neighborId]);
                    if (candidates.size() < ef || dis < lowerBound)
                    {
                        candidates.emplace(dis, neighborId);
                        pool.emplace(dis, neighborId);

                        if (candidates.size() > ef)
                        {
                            candidates.pop();
                        }
                        if (candidates.size())
                        {
                            lowerBound = candidates.top().first;
                        }
                    }
                }
            }

            while (candidates.size() > query_k)
                candidates.pop();
            return candidates;
        }

        std::vector<PFI> PruneByHeuristic2(std::vector<PFI> &old_list, std::vector<PFI> &new_list)
        {
            std::priority_queue<PFI, std::vector<PFI>, std::greater<PFI>> queue_closest;
            std::vector<PFI> return_list;
            std::vector<bool> return_list_belong_to_lowerlayer_list;
            for (auto t : old_list)
                queue_closest.emplace(t);
            for (auto t : new_list)
                queue_closest.emplace(t);
            if (queue_closest.size() <= M)
            {
                while (queue_closest.size())
                {
                    return_list.emplace_back(queue_closest.top());
                    queue_closest.pop();
                }
                return return_list;
            }

            while (queue_closest.size())
            {
                if (return_list.size() >= M)
                    break;

                auto current_pair = queue_closest.top();
                float dist_to_pid = current_pair.first;
                queue_closest.pop();

                bool good = true;
                bool current_old = false;
                for (auto t : old_list)
                {
                    if (t.second == current_pair.second)
                    {
                        current_old = true;
                        break;
                    }
                }
                for (int i = 0; i < return_list.size(); i++)
                {
                    if (current_old && return_list_belong_to_lowerlayer_list[i])
                        continue;
                    auto second_pair = return_list[i];
                    float curdist = dis_compute(storage->data_points[current_pair.second], storage->data_points[second_pair.second]);
                    if (curdist < dist_to_pid)
                    {
                        good = false;
                        break;
                    }
                }
                if (good)
                {
                    return_list.emplace_back(current_pair);
                    return_list_belong_to_lowerlayer_list.emplace_back(current_old);
                }
            }
            return return_list;
        }

        void process_node(TreeNode *u)
        {
            if (u->childs.size() == 0)
                return;

            copyfirstchild(u);
            int merged_point_num = u->childs[0]->rbound - u->childs[0]->lbound + 1;
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);

            for (int i = 1; i < u->childs.size(); i++)
            {
                std::uniform_int_distribution<int> u_start(0, merged_point_num - 1);
                TreeNode *cur_child = u->childs[i];
                for (int pid = cur_child->lbound; pid <= cur_child->rbound; pid++)
                {
                    std::vector<int> enterpoints;
                    for (int i = 0; i < std::min(3, merged_point_num); i++)
                    {
                        int enterpid = u_start(e) + u->lbound;
                        enterpoints.emplace_back(enterpid);
                    }
                    // enterpoints.emplace_back(u->lbound);
                    // enterpoints.emplace_back(u->lbound + (merged_point_num - 1) / 2);
                    // enterpoints.emplace_back(u->lbound + merged_point_num - 1);
                    auto search_result = search_on_incomplete_graph(u, storage->data_points[pid], ef_construction, ef_construction, enterpoints);
                    while (search_result.size())
                    {
                        edges[pid][mapLayer(u->depth)].emplace_back(search_result.top());
                        search_result.pop();
                    }
                    edges[pid][mapLayer(u->depth)] = PruneByHeuristic2(edges[pid][mapLayer(cur_child->depth)], edges[pid][mapLayer(u->depth)]);
                }

                for (int j = 0; j < merged_point_num; j++)
                {
                    int pid = u->lbound + j;
                    reverse_edges[pid].clear();
                }

                for (int pid = cur_child->lbound; pid <= cur_child->rbound; pid++)
                {
                    for (auto neighbor_pair : edges[pid][mapLayer(u->depth)])
                    {
                        int neighborId = neighbor_pair.second;
                        if (neighborId < cur_child->lbound)
                        {
                            reverse_edges[neighborId].emplace_back(neighbor_pair.first, pid);
                        }
                    }
                }

                for (int j = 0; j < merged_point_num; j++)
                {
                    int pid = u->lbound + j;
                    edges[pid][mapLayer(u->depth)] = PruneByHeuristic2(edges[pid][mapLayer(u->depth)], reverse_edges[pid]);
                }

                merged_point_num += cur_child->rbound - cur_child->lbound + 1;
            }
        }

        void buildindex()
        {
            reverse_edges.resize(storage->data_nb);
            std::vector<std::vector<TreeNode *>> level_nodes;
            level_nodes.resize(max_depth + 1);
            for (auto node : tree->treenodes)
            {
                level_nodes[mapLayer(node->depth)].emplace_back(node);
            }
            for (int layer = max_depth; layer >= 0; layer--)
            {
                int _layer = mapLayer(layer);
                // std::cout << "building for layer " << _layer << std::endl;
                std::vector<std::thread> threads;

                for (int i = 0; i < level_nodes[_layer].size(); i++)
                {
                    if (threads.size() >= max_threads)
                    {
                        for (auto &thread : threads)
                        {
                            if (thread.joinable())
                            {
                                thread.join();
                            }
                        }
                        threads.clear();
                    }
                    TreeNode *u = level_nodes[_layer][i];
                    threads.emplace_back(std::thread(&iRangeGraph_Build<dist_t>::process_node, this, u));
                }
                for (auto &thread : threads)
                {
                    if (thread.joinable())
                    {
                        thread.join();
                    }
                }

                threads.clear();
            }
        }

        void buildandsave(std::string indexpath)
        {
            // CheckPath(indexpath);
            std::ofstream indexfile(indexpath, std::ios::out | std::ios::binary);
            if (!indexfile.is_open())
                throw Exception("cannot open " + indexpath);
            reverse_edges.resize(storage->data_nb);
            timeval t1, t2;
            gettimeofday(&t1, NULL);
            buildindex();
            gettimeofday(&t2, NULL);
            double construction_time = GetTime(t1, t2);

            std::cout << "construction time:" << construction_time << "s" << std::endl;

            for (int pid = 0; pid < storage->data_nb; pid++)
            {
                for (int layer = 0; layer <= tree->max_depth; layer++)
                {
                    int size = edges[pid][layer].size();
                    indexfile.write((char *)&size, sizeof(int));
                    for (int i = 0; i < size; i++)
                    {
                        int neighborId = edges[pid][layer][i].second;
                        indexfile.write((char *)&neighborId, sizeof(int));
                    }
                }
            }

            std::cout << "save index done" << std::endl;
            indexfile.close();
        }
        void save(std::string indexpath){
            std::ofstream indexfile(indexpath, std::ios::out | std::ios::binary);
            if (!indexfile.is_open())
                throw Exception("cannot open " + indexpath);
            for (int pid = 0; pid < storage->data_nb; pid++)
            {
                for (int layer = 0; layer <= max_depth; layer++)
                {
                    int size = edges[pid][layer].size();
                    indexfile.write((char *)&size, sizeof(int));
                    for (int i = 0; i < size; i++)
                    {
                        int neighborId = edges[pid][layer][i].second;
                        indexfile.write((char *)&neighborId, sizeof(int));
                    }
                }
            }
        }
        void SetStorage(DataLoader *_storage){
            storage = _storage;
        }


        void MergeEdges(iRangeGraph_Build<dist_t> *_builder){
            assert(max_depth + 1 == edges[0].size());
            std::vector<std::vector<std::vector<std::pair<float, int>>>> _edges = _builder->edges;
            int offset = edges.size(), data_nb = offset, _data_nb = _edges.size();


            //拷贝visitedpool
            visitedpool.resize(data_nb + _data_nb);

            //把新的边已有的edges和点拷贝过来
            if(data_nb + _data_nb != storage->data_nb){
                std::cout << "index1 has " << data_nb << " datas" << std::endl;
                std::cout << "index2 has " << data_nb << " datas" << std::endl;
                std::cout << "storage has " << storage->data_nb << " datas" << std::endl; 
                throw Exception("Data_nb is wrong");
            }
            if(data_nb != _data_nb){
                throw Exception("Try to merge two index with different layers");
            }

            // std::cout << "copyfirstchild done" << std::endl;

            //将_edges的边进行偏移
            //首先将等大小的新索引添加进来，保证除了最高层，其余的添加完毕
            edges.reserve(data_nb + _data_nb);
            for(int id = 0; id < _data_nb; id++){
                //添加点i的边
                assert(_edges[id].size() > max_depth);
                for(int layer = 0; layer <= max_depth; layer++){
                    //添加点i第j层的边 + offset
                    for(int neighbor = 0; neighbor < _edges[id][layer].size(); neighbor++){
                        _edges[id][layer][neighbor] = {_edges[id][layer][neighbor].first, _edges[id][layer][neighbor].second + offset};
                    }
                }
            }

            
            //把偏移后的除最高层外的边加入到edges中
            for(int i = 0; i < _data_nb; i++){
                edges.emplace_back(_edges[i]);
                assert(edges.back().size() > 0);
                assert((int)edges.size() == data_nb + i + 1);
            }


            //copyfirstchild
            for(int id = 0; id < data_nb; id++){
                if (edges[id].size() <= static_cast<std::size_t>(max_depth)) {
                    std::ostringstream oss;
                    oss << "edges[" << id << "].size() <= max_depth (" << edges[id].size()
                        << " <= " << max_depth << ")";
                    throw Exception(oss.str());
                }
                edges[id].reserve(edges[id].size() + 1);
                edges[id].emplace_back(edges[id][max_depth]);
            }
            max_depth++;
            //建新的一层的边
            int layer = max_depth;

            //检查边是否合理
            for(int id = 0; id < data_nb + _data_nb; id++){
                //添加点i的边
                assert(id < (int)edges.size());
                for(int layer = 0; layer < max_depth; layer++){
                    //添加点i第j层的边 + offset
                    for(int neighbor = 0; neighbor < edges[id][layer].size(); neighbor++){
                        int bound = id < data_nb ? 0 : 0 - data_nb;
                        assert(edges[id][layer][neighbor].second + bound < data_nb);
                    }
                }
            }            

// ---------------------------------------------------------------------------
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);
            std::vector<std::vector<PFI>> reverse_edges(data_nb);
            TreeNode *u = new TreeNode(0, 0, 0);
            for(int i = data_nb; i < data_nb + _data_nb; i++){
                //为点i选邻居
                // assert(i < storage->data_points.size());
                // assert(i < (int)edges.size());
                
                // assert(u != nullptr);
                //生成entry_points
                std::vector<int> enterpoints;

                int l_bound = 0;
                int r_bound = data_nb - 1;
                std::uniform_int_distribution<int> u_start(l_bound, r_bound);
                for(int j = 0; j < 3; j++){
                    int enterpid = u_start(e);
                    assert(enterpid >= 0 && enterpid < data_nb);
                    enterpoints.emplace_back(enterpid);
                }

                auto search_result = search_on_incomplete_graph(u, storage->data_points[i], ef_construction ,ef_construction, enterpoints);
                //在另外的子段查询邻居

                // std::priority_queue<PFI> search_result;
                // for(int j = 0; j < 3; j++)
                // {search_result.emplace(dis_compute(storage->data_points[i], storage->data_points[enterpoints[j]]), enterpoints[j]);}
                
                
                while(search_result.size()){
                    //遍历查询结果
                    PFI edge = search_result.top();
                    int neighborId = edge.second;
                    if(neighborId >= data_nb || neighborId < 0){
                        throw Exception("Search wrong");
                    }
                    assert(reverse_edges.size() > neighborId);
                    edges[i].resize(max_depth + 1);
                    edges[i][max_depth].emplace_back(edge);
                    search_result.pop();
                }
                edges[i][max_depth] = PruneByHeuristic2(edges[i][max_depth - 1], edges[i][max_depth]);
                int edge_nb = edges[i][max_depth].size();
                for(int l = 0; l < edge_nb; l++){
                    PFI edge = edges[i][max_depth][l];
                    if(edge.second < data_nb){
                        reverse_edges[edge.second].push_back({edge.first, i});
                    }
                }
           }

            //using the reverse edges and the edges of lower layer to construct the edges of the max_depth layer
            for(int i = 0; i < data_nb; i++){
                edges[i][max_depth] = PruneByHeuristic2(edges[i][max_depth], reverse_edges[i]);
            }
            assert(max_depth + 1 == edges[0].size());

        }
    };

}
