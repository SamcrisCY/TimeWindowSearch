#pragma once
#define MAX_TEN 90000
#include <thread>
#include <map>
#include <queue>
#include <random>
#include <chrono>
#include <fstream>

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "utils.h"
#include "searcher.hpp"
#include "memory.hpp"
#include "construction.h"
#include "iRG_search.h"



namespace TimeWindowIndex
{

    typedef std::pair<float, int> PFI;
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;
    typedef int bucketId;
    //判断当前的search是否可以直接使用

    // Single unified class that contains both build and search functionality.
    template <typename dist_t>
    class iRangeGraph
    {
        public:
        //storage contains the vector datas, query datas and paremeters, query ranges and groundtruths
            ::iRangeGraph::DataLoader *storage;
            ::iRangeGraph::iRangeGraph_Build<dist_t> *builder;
            ::iRangeGraph::iRangeGraph_Search<dist_t> *searcher;
            std::unordered_map<int, int> id_hash;//前面是internal_id，后面是整个数据库的id
            int M;
            int ef_constrction;
            std::string index_path;
            bool need_refresh = false;
            int max_depth = 0;
            iRangeGraph(::iRangeGraph::DataLoader *_storage, int _M, int _ef_construction)
                : storage(_storage), M(_M), ef_constrction(_ef_construction)
            {
                builder = new ::iRangeGraph::iRangeGraph_Build<dist_t>(_storage, _M, ef_constrction);
                builder->max_threads = 1;
                timeval t1, t2;

                gettimeofday(&t1, NULL);
                builder->buildindex();
                gettimeofday(&t2, NULL);
                double construction_time = GetTime(t1, t2);
                // this->edges = builder->edges;
                searcher = new ::iRangeGraph::iRangeGraph_Search<dist_t>(builder->edges, _storage, _M);
                max_depth = builder->edges.front().size();
            }
            void refresh(){
                if(need_refresh){
                    // this->edges = builder->edges;
                    searcher = new ::iRangeGraph::iRangeGraph_Search<dist_t>(builder->edges, storage, M);
                }
                need_refresh = false;
            }

            void save(std::string filename){
                builder->save(filename);
            }

            void LoadQuery(std::string filename){
                this->storage->LoadQuery(filename);
            }
            void LoadQueryRange(std::string filename){
                this->storage->LoadQueryRange(filename);
            }
            void LoadGroundtruth(std::string filename){
                this->storage->LoadGroundtruth(filename);
            }
            int getListCount(linklistsizeint *ptr) const
            {
                return *((int *)ptr);
            }
            void Search(std::vector<int> &SearchEF, std::string saveprefix){
                refresh();
                searcher->search(SearchEF, saveprefix, M);
            }

            std::vector<PFI> Search(int ef, int l, int r, std::vector<float> query_vector, bool _use_fusion = false, double ten = -1){
                refresh();
                return searcher->search(ef, M, l, r, query_vector, _use_fusion, ten);
            }

            void MergeIndex(iRangeGraph *index_ex){
                //storage merge
                storage->MergeStorage(index_ex->storage);
                //builders edges merge
                builder->MergeEdges(index_ex->builder);
                // builder->edges = builder_e.edges;
                need_refresh = true; 
                max_depth++;
                // delete index_ex;
            }
            ~iRangeGraph() {
                delete storage;
                delete builder;
                delete searcher;
            }

    }; 

    template <typename dist_t>
    class LiveIndex{
        public:
            int M, ef_construction;
            EuclideanSquareDistance discalculator;
            std::unordered_map<int, std::shared_ptr<iRangeGraph<dist_t>>> baseindex;//作为插入操作后的
            std::vector<VectorDataType> buffer;
            std::vector<double> fusion_buffer;
            int buffer_max;
            std::vector<double> t_list;
            int Dim = 0;
            LiveIndex(int _M, int _ef_construction, int _Dim,  int _buffer_max = -1): M(_M), ef_construction(_ef_construction), Dim(_Dim){
                if(_buffer_max == -1){
                    buffer_max = 2 * _M;
                }else{
                    buffer_max = _buffer_max;
                }
            }
            void Refresh(){
                for(auto pair: baseindex){
                    pair.second->refresh();
                }
            }
            
            std::priority_queue<PFI> Search(int ef, double qten, VectorDataType query_vector, int query_K, bool use_fusion = false){
                std::priority_queue<PFI> ans;
                for(auto v: buffer){
                    ans.push({discalculator(v, query_vector), v.vid});
                }
                //live index search
                //t < qten
                timeval t1, t2;
                double searchtime = 0.0;
                for(auto pair: baseindex){
                    std::shared_ptr<iRangeGraph<dist_t>> index = pair.second;
                    if(index->storage->data_nb == index->storage->tomb_nb){
                        std::cout << "this index does not have any live data" << std::endl;
                        continue;
                        //stands for this live index do not have any live data
                    }
                    std::pair<double, double> index_range = index->storage->GetTBound();
                    
                    if(index_range.first < qten){
                        int r = index->storage->data_nb - 1;
                        int l = 0;
                        if(true){
                            //live index 
                            // u.tst <= q.ten
                            if(index_range.second > qten){
                                r = std::upper_bound(index->storage->t_hash.begin(), index->storage->t_hash.end(), qten)
                                 - index->storage->t_hash.begin() - 1; 
                            }else{
                            }
                        }
                        
                        std::vector<PFI> res;
                        res = index->Search(ef, l, r, query_vector.data);
                        for(PFI r : res){
                            ans.push(r);
                        }
                        while(ans.size() > ef){
                            ans.pop();
                        }
                    }
                }
                return ans;
            }

            std::priority_queue<PFI> Search(int ef, double qtst, double qten, VectorDataType query_vector, int query_K, bool use_fusion = false){
                std::priority_queue<PFI> ans;
                //dead index search
                for(auto v: buffer){
                    ans.push({discalculator(v, query_vector), v.vid});
                }
                // timeval t1, t2;
                // double searchtime = 0.0;
                for(auto pair: baseindex){
                    std::shared_ptr<iRangeGraph<dist_t>> index = pair.second;
                    std::pair<double, double> index_range = index->storage->GetTBound();
                    if(index_range.second >= qtst){
                        //dead index use ten qtst
                        // std::cout << "Index Range [" << index_range.first << ", " << index_range.second << "]" << std::endl;
                        int r = index->storage->data_nb - 1;
                        int l = 0;
                        //dead index
                        // u.ten >= q.tst
                        if(index_range.first < qtst){
                            l = std::lower_bound(index->storage->t_hash.begin(), index->storage->t_hash.end(), qtst)
                              - index->storage->t_hash.begin() + 1;
                        }
                        // std::cout << "ID Range [" << l << ", " << r << "]" << std::endl;

                        std::vector<PFI> res = index->Search(ef, l, r, query_vector.data, use_fusion, qten);
                        for(PFI r : res){
                            ans.push(r);
                        }
                        // while(ans.size() > ef){
                        //     ans.pop();
                        // }
                    }else{
                        // std::cout << "Index Range [" << index_range.first << ", " << index_range.second << "]" << std::endl;
                        // std::cout << "Query Range [" << qtst << ", " << ten << "]\n";
                    }
                    // std::cout << ans.size() << std::endl;

                }
                return ans;
            }

            std::pair<int, int> Insert(std::vector<std::pair<double, VectorDataType>> insert_vectors){
                int insert_num = insert_vectors.size();
                double upbound = -1;
                double lowbound = 1e8;
                for(int i = 0; i < insert_num; i++){
                    buffer.emplace_back(insert_vectors[i].second);
                    int t = insert_vectors[i].first;
                    t_list.emplace_back(t);
                    upbound = upbound > t ? upbound : t;
                    lowbound = lowbound < t ? lowbound : t;
                }
                //加入buffer，如果插入到达最大值时，构建新的iRangeGraph并插入到backup中
                ClearBuffer();
                return {upbound, lowbound};
            }

            std::pair<int, int> Insert(std::vector<std::pair<double, VectorDataType>> insert_vectors, std::vector<double> tst_list){
                int insert_num = insert_vectors.size();
                double upbound = -1;
                double lowbound = 1e8;
                for(int i = 0; i < insert_num; i++){
                    buffer.emplace_back(insert_vectors[i].second);
                    int t = insert_vectors[i].first;
                    t_list.emplace_back(t);
                    fusion_buffer.push_back(tst_list[i]);
                    upbound = upbound > t ? upbound : t;
                    lowbound = lowbound < t ? lowbound : t;
                }
                //加入buffer，如果插入到达最大值时，构建新的iRangeGraph并插入到backup中
                ClearBuffer();
                return {upbound, lowbound};
            }

            int getBufferId(int remove_id){
                int l = 0, r = buffer.size() - 1;
                while(l <= r){
                    int mid = l + (r - l) / 2;
                    if(buffer[mid].vid < remove_id){
                        l = mid + 1;      
                    } else if(buffer[mid].vid > remove_id){
                        r = mid - 1;       
                    } else {
                        return mid;       
                    }
                }
                return -1; 
            }

            void checkId(){
                for(auto pair: baseindex){
                    auto index = pair.second;

                }
            }

            void PrintRange(){
                for(auto pair: baseindex){
                    auto index = pair.second;
                    std::cout << "[" << index->storage->GetIdBound().second << ", " << index->storage->GetIdBound().first << "] \n"; 
                }
            }
            void Remove(std::vector<std::pair<double, VectorDataType>> remove_ids){
                //添加墓碑标记，判断删除后段是否为空
                int remove_nb = remove_ids.size();
                std::vector<VectorDataType> remove_vectors;
                for(int i = 0; i < remove_nb; i++){
                    double ten = remove_ids[i].first;
                    int remove_id = remove_ids[i].second.vid;
                    //Get iRangeGraph * by Globol Id and remove data
                    for(auto record : baseindex){
                        std::shared_ptr<iRangeGraph<dist_t>> irg = record.second;
                        if(irg->storage->GetIdBound().first >= remove_id
                        && irg->storage->GetIdBound().second <= remove_id){
                            irg->storage->RemoveData(remove_id);
                            if(irg->storage->tomb_nb == irg->storage->data_nb){
                                baseindex.erase(record.first);
                                // std::cout << "Remove index with range [" << irg->storage->GetIdBound().second
                                //  << ", " << irg->storage->GetIdBound().first<< "]\n";
                            }
                            break;
                        }
                    }
                    if(!buffer.empty() && remove_id >= buffer[0].vid){
                        int bufferid = getBufferId(remove_id);
                        if(buffer[bufferid].vid != remove_id){
                            throw Exception("Wrong buffer Internal Id:" + std::to_string(bufferid) + " " + std::to_string(remove_id) + " " + std::to_string(ten));
                        }
                        buffer.erase(buffer.begin() + bufferid);
                    }                                                                                                                                                                                                                                                                                                                                                                                            
                }
            }

            void ClearBuffer(){
                int buffer_size = buffer.size();
                if(buffer_size < buffer_max){
                    return;
                }
                int max_depth = static_cast<int>(std::floor(std::log2(buffer_size))) + 1;
                int load_nb = 1 << (max_depth - 1);
                ::iRangeGraph::DataLoader *storage = new ::iRangeGraph::DataLoader();
                // for(int i = 0; i < load_nb; i++){
                //     storage->AddData(buffer[i], t_list[i]);
                // }
                if(fusion_buffer.empty()){
                    for(int i = 0; i < load_nb; i++){
                        storage->AddData(buffer[i], t_list[i]);
                    }                    
                }else{
                    for(int i = 0; i < load_nb; i++){
                        storage->AddData(buffer[i], t_list[i], fusion_buffer[i]);
                    }
                    fusion_buffer.clear();                        
                }
                //清空buffer中前load_nb个数据
                std::vector<VectorDataType> vbackup(buffer.begin() + load_nb, buffer.end());
                buffer.clear();
                buffer.swap(vbackup);
                std::vector<double> tbackup(t_list.begin() + load_nb, t_list.end());
                t_list.clear();
                t_list.swap(tbackup);
                //构造iRangeGraph  
                std::shared_ptr<iRangeGraph<dist_t>> irg = std::make_shared<iRangeGraph<dist_t>>(storage, M, ef_construction);
                AddIndex(irg);
            }
             void AddIndex(std::shared_ptr<iRangeGraph<dist_t>> irg){

                int depth = irg->max_depth;
                while(baseindex.find(depth) != baseindex.end()){
                    baseindex[depth]->MergeIndex(irg.get());
                    irg = baseindex[depth];
                    baseindex.erase(depth);
                    depth++;
                }
                baseindex[depth]=irg;
            }
            void Save(std::string filename){
                //todo
            }
            void Remake(){
                std::vector<int> depth_list;
                for(auto pair: baseindex){
                    depth_list.push_back(pair.first);
                }
                for(int j = 0; j < depth_list.size(); j++){
                    
                    std::shared_ptr<::iRangeGraph::DataLoader> st = std::make_shared<::iRangeGraph::DataLoader>();
                    std::shared_ptr<iRangeGraph<dist_t>> irg = baseindex[depth_list[j]];
                    int n = irg->storage->data_nb;
                    for(int i = 0; i < n; i++){
                        VectorDataType v(irg->storage->Dim, irg->storage->vid_hash[i].first, irg->storage->data_points[i]);
                        st->AddData(v, irg->storage->t_hash[i]);
                    }
                    std::shared_ptr<iRangeGraph<dist_t>> new_irg = std::make_shared<iRangeGraph<dist_t>>(st.get(), M, ef_construction);
                    baseindex[depth_list[j]] = new_irg;
                }                
            }

    };
    
    template <typename dist_t>
    class BucketIndex{
        public:
            double tst_upbound;
            double tst_lowbound;
            LiveIndex<dist_t> *index;
            int Dim;
            BucketIndex(int _M, int _ef_construction, int _Dim,  int _buffer_max = -1): Dim(_Dim)
            {
                index = new LiveIndex<dist_t>(_M, _ef_construction, _Dim, _buffer_max);
            }
            void Insert(std::vector<std::pair<double, VectorDataType>> insert_vectors, std::vector<double> tst_list){
                std::pair<int, int> insert_timew = index->Insert(insert_vectors, tst_list);
                UpdateBound(insert_timew);
            }
            void UpdateBound(std::pair<int, int> new_timewindow){
                int upbound = new_timewindow.first;
                int lowbound = new_timewindow.second;
                tst_upbound = tst_upbound > upbound ? tst_upbound : upbound;
                tst_lowbound = tst_lowbound < lowbound ? tst_lowbound : lowbound;
            }
            int Overlap(double ten){
                //1 stands for having overlap but not subset
                //2 stands for subset
                //3 stands for no overlap
                if(ten >= tst_upbound){
                    return 2;
                }else if(ten <= tst_upbound && ten >= tst_lowbound){
                    return 1;
                }else{
                    return 3;
                }
            }
    };

    template <typename dist_t>
    class DeadIndex{
        public:
            int M, ef_construction;
            bool use_bucket;
            int tst_upbound, tst_lowbound;
            ::iRangeGraph::DataLoader *storage;
            std::vector<VectorDataType> buffer;
            int buffer_max;
            std::vector<BucketIndex<dist_t> *> baseindex;
            int Dim;
            //if use_bucket, baseindex has one member {-1, index*}
            //else, baseindex has some members which are associated with a bucket id
            //每三个小时有一个bucket，一共有8个bucket
            DeadIndex(int _M, int _ef_construction, int _buffer_max, int _Dim, bool _use_bucket = true)
             : M(_M), ef_construction(_ef_construction), use_bucket(_use_bucket), Dim(_Dim){
                if(_buffer_max == -1){
                    buffer_max = 2 * _M;
                }else{
                    buffer_max = _buffer_max;
                }
                if(!_use_bucket){
                    BucketIndex<dist_t> *bindex = new BucketIndex<dist_t>(_M, _ef_construction, _Dim, _buffer_max);
                    baseindex.emplace_back(bindex);
                }else{
                    for(int i = 0; i < 4; i++){
                        BucketIndex<dist_t> *bindex = new BucketIndex<dist_t>(_M, _ef_construction, _Dim, _buffer_max);
                        baseindex.emplace_back(bindex);                        
                    }
                }
                storage = new ::iRangeGraph::DataLoader();
            }
            void Insert(std::vector<std::pair<double, VectorDataType>> insert_vectors, std::vector<double> tst_list){
                //ten用来建索引，tst用来分bucket和计算distance fusion
                if(!use_bucket){
                    //说明只有一个
                    baseindex[0]->Insert(insert_vectors, tst_list);
                }else{

                    double t = tst_list[0];
                    baseindex[int(t / (6 * 3600))]->Insert(insert_vectors, tst_list);
                }
            }
            void Refresh(){
                for(auto bi: baseindex){
                    bi->index->Refresh();
                }
            }
            void PrintRange(){
                for(auto bi: baseindex){
                    bi->index->PrintRange();
                }
            }
            std::priority_queue<PFI> Search(int ef, double tst, double ten, VectorDataType query_vector, int query_K){
                std::priority_queue<PFI> ans;
                std::priority_queue<PFI> res;
                for(auto index: baseindex){
                    if(index->Overlap(ten) == 1){
                        res = index->index->Search(ef, tst, ten, query_vector, query_K, false);
                    }else if(index->Overlap(ten) == 2){
                        res = index->index->Search(ef, tst, ten, query_vector, query_K, false);
                    }
                    while(!res.empty()){
                        ans.push(res.top());

                        res.pop();
                    }
                }
                while(ans.size() > ef){
                    ans.pop();
                }

                return ans;           
            }
    };
} 
