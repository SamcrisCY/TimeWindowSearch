#pragma once

#include "space_l2.h"
#include <filesystem>
#include <string>
#include <cstring>
#include <vector>
#include <fstream>
#include <sys/time.h>
#include <map>
#include "./utils/DataType.hpp"
#include "./utils/MetricType.hpp"
#include "./utils/BenchLogger.hpp"
#include "./utils/File_IO.h"

class Exception : public std::runtime_error
{
public:
    Exception(const std::string &msg) : std::runtime_error(msg) {}
};

void CheckPath(std::string filename)
{
    std::filesystem::path pathObj(filename);
    std::filesystem::path dirPath = pathObj.parent_path();
    if (!std::filesystem::exists(dirPath))
    {
        try
        {
            if (std::filesystem::create_directories(dirPath))
            {
                std::cout << "Directory created: " << dirPath << std::endl;
            }
            else
            {
                std::cerr << "Failed to create directory: " << dirPath << std::endl;
            }
        }
        catch (std::filesystem::filesystem_error &e)
        {
            throw Exception(e.what());
        }
    }
}

float GetTime(timeval &begin, timeval &end)
{
    return end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) * 1.0 / CLOCKS_PER_SEC;
}

namespace iRangeGraph
{
    typedef std::pair<float, int> PFI;
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;
    typedef int internalId;
    typedef int globalId;
    class DataLoader
    {
    public:
        int Dim, query_nb, query_K;
        std::vector<std::vector<float>> query_points;
        int data_nb = 0, tomb_nb = 0;
        std::vector<std::vector<float>> data_points;
        std::unordered_map<int, std::vector<std::pair<int, int>>> query_range;
        std::unordered_map<int, std::vector<std::vector<int>>> groundtruth;
        std::unordered_map<internalId, std::pair<globalId, bool>> vid_hash;
        std::vector<double> t_hash;
        std::vector<double> fusion_hash;

        bool operator==(const DataLoader& other) const{
            return Dim == other.Dim && data_nb == other.data_nb && data_points == other.data_points
            && vid_hash == other.vid_hash && t_hash == other.t_hash;
        }
        DataLoader() {}
        // ~DataLoader() {}
        std::pair<int, int> GetIdBound(){
            return {vid_hash[data_nb - 1].first, vid_hash[0].first};
        }
        std::pair<float, float> GetTBound(){
            return {t_hash[0], t_hash[data_nb - 1]};
        }
        bool Overlap(std::pair<double, double> range1, std::pair<double, double> range2){
            int l1 = range1.first;
            int r1 = range1.second;
            int l2 = range2.first;
            int r2 = range2.second;
            if(l2 <= r1 && l1 >= r2){
                return true;
            }
            return false;
        }
        std::vector<float> RemoveData(int remove_id){
            int offset = vid_hash[0].first;
            int internalId = std::min(remove_id - offset, data_nb - 1);
            while(vid_hash[internalId].first != remove_id && internalId >= 0){
                internalId--;
            }
            if(vid_hash[internalId].second){
                throw Exception("Repeatly Remove " + std::to_string(remove_id));
            }
            vid_hash[internalId] = {remove_id, true};
            tomb_nb++;
            return data_points[internalId];
        }
        // query vector filename format: 4 bytes: query number; 4 bytes: dimension; query_nb*Dim vectors
        void LoadQuery(std::string filename)
        {
            std::ifstream infile(filename, std::ios::in | std::ios::binary);
            if (!infile.is_open())
                throw Exception("cannot open " + filename);
            infile.read((char *)&query_nb, sizeof(int));
            infile.read((char *)&Dim, sizeof(int));
            query_points.resize(query_nb);
            for (int i = 0; i < query_nb; i++)
            {
                query_points[i].resize(Dim);
                infile.read((char *)query_points[i].data(), Dim * sizeof(float));
            }
            infile.close();
        }

        void MergeStorage(DataLoader *_storage){
            //merge data_points and vid_hash
            // std::vector<std::vector<float>> _data_points = _storage->data_points;
            // std::unordered_map<internalId, std::pair<globalId, bool>> _vid_hash = _storage->vid_hash;
            int data_nb = data_points.size();
            int _data_nb = _storage->data_points.size();

            for(int i = 0; i < _data_nb; i++){
                data_points.emplace_back(_storage->data_points[i]);
                vid_hash.insert({i + data_nb, _storage->vid_hash[i]});
                t_hash.emplace_back(_storage->t_hash[i]);
                if(!_storage->fusion_hash.empty()){
                    fusion_hash.emplace_back(_storage->fusion_hash[i]);
                }
            }
            this->data_nb += _data_nb;
            this->tomb_nb += _storage->tomb_nb;
            assert(this->data_nb == data_points.size());
            // delete(_storage);
        }
        void LoadData(std::string filename)
        {
            std::vector<VectorDataType> vector_list;
            ReadVectorData(filename, vector_list);
            std::cout << "Load " << vector_list.size() << " Datas" << std::endl;
            data_nb = vector_list.size();
            Dim = vector_list[0].Dimension();
            data_points.resize(data_nb);
            for(int i = 0; i < data_nb; i++){
                data_points[i] = vector_list[i].data;
                vid_hash.insert({i, {vector_list[i].vid, false}});
                t_hash.emplace_back(0.0);
            }
        }
        void AddData(VectorDataType v, double t){
            Dim = v.Dimension();
            data_nb = data_points.size();
            data_points.emplace_back(v.data);
            vid_hash.insert({data_nb, {v.vid, false}});
            t_hash.emplace_back(t);
            this->data_nb += 1;
        }
        void AddData(VectorDataType v, double ten, double tst){
            Dim = v.Dimension();
            data_nb = data_points.size();
            data_points.emplace_back(v.data);
            vid_hash.insert({data_nb, {v.vid, false}});
            t_hash.emplace_back(ten);
            this->data_nb += 1;
            AddFusion(tst);
        }
        void AddFusion(double tst){
            fusion_hash.push_back(tst);
        }
        // void LoadData(std::string filename)
        // {
        //     std::ifstream infile(filename, std::ios::in | std::ios::binary);
        //     if (!infile.is_open())
        //         throw Exception("cannot open " + filename);
        //     infile.read((char *)&data_nb, sizeof(int));
        //     infile.read((char *)&Dim, sizeof(int));
        //     data_points.resize(data_nb);
        //     for (int i = 0; i < data_nb; i++)
        //     {
        //         data_points[i].resize(Dim);
        //         infile.read((char *)data_points[i].data(), Dim * sizeof(float));
        //     }
        //     infile.close();
        // }
        void LoadData(std::vector<VectorDataType> vector_list){
            data_nb = vector_list.size();
            Dim = vector_list[0].Dimension();
            data_points.resize(data_nb);
            for(int i = 0; i < data_nb; i++){
                data_points[i] = vector_list[i].data;
                vid_hash.insert({i, {vector_list[i].vid, false}});
            }
        }

        // By default generation, 0.bin~9.bin denotes 2^0~2^-9 range fractions, 17.bin denotes mixed range fraction.
        // Before reading the query ranges, make sure query vectors have been read.
        void LoadQueryRange(std::string fileprefix)
        {
            std::vector<int> s;
            for (int i = 0; i < 10; i++)
                s.emplace_back(i);
            s.emplace_back(17);
            for (auto suffix : s)
            {
                std::string filename = fileprefix + std::to_string(suffix) + ".bin";
                std::ifstream infile(filename, std::ios::in | std::ios::binary);
                if (!infile.is_open())
                    throw Exception("cannot open " + filename);
                for (int i = 0; i < query_nb; i++)
                {
                    int ql, qr;
                    infile.read((char *)&ql, sizeof(int));
                    infile.read((char *)&qr, sizeof(int));
                    query_range[suffix].emplace_back(ql, qr);
                }
                infile.close();
            }
        }
        //todo 修改queryRange形式

        // 0.bin~9.bin correspond to groundtruth for 2^0~2^-9 range fractions, 17.bin for mixed fraction
        void LoadGroundtruth(std::string fileprefix)
        {
            for (auto t : query_range)
            {
                int suffix = t.first;
                std::string filename = fileprefix + std::to_string(suffix) + ".bin";
                std::ifstream infile(filename, std::ios::in | std::ios::binary);
                if (!infile.is_open())
                    throw Exception("cannot open " + filename);
                groundtruth[suffix].resize(query_nb);
                for (int i = 0; i < query_nb; i++)
                {
                    groundtruth[suffix][i].resize(query_K);
                    infile.read((char *)groundtruth[suffix][i].data(), query_K * sizeof(int));
                }
                infile.close();
            }
        }
    };

    class QueryGenerator
    {
    public:
        int data_nb, query_nb;
        hnswlib::L2Space *space;

        QueryGenerator(int data_num, int query_num) : data_nb(data_num), query_nb(query_num) {}
        ~QueryGenerator() {}

        void GenerateRange(std::string saveprefix)
        {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);

            std::vector<std::pair<int, int>> rs;
            int current_len = data_nb;
            for (int i = 0; i < 10; i++)
            {
                if (current_len < 10)
                    throw Exception("dataset size is too small, increase the amount of data objects!");
                rs.emplace_back(current_len, i);
                current_len /= 2;
            }
            for (auto t : rs)
            {
                int len = t.first, suffix = t.second;
                std::string savepath = saveprefix + std::to_string(suffix) + ".bin";
                // CheckPath(savepath);
                std::cout << "save query range to" << savepath << std::endl;
                std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);
                std::uniform_int_distribution<int> u_start(0, data_nb - len);
                for (int i = 0; i < query_nb; i++)
                {
                    int ql = u_start(e);
                    int qr = ql + len - 1;
                    if (ql >= data_nb || qr >= data_nb)
                        throw Exception("Query range out of bound");
                    outfile.write((char *)&ql, sizeof(int));
                    outfile.write((char *)&qr, sizeof(int));
                }
                outfile.close();
            }

            rs.clear();
            current_len = data_nb;
            for (int i = 0; i < 10; i++)
            {
                rs.emplace_back(current_len, i);
                current_len /= 2;
            }
            std::string savepath = saveprefix + "17.bin";
            // CheckPath(savepath);
            std::cout << "save query range to" << savepath << std::endl;
            std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
            if (!outfile.is_open())
                throw Exception("cannot open " + savepath);

            for (auto t : rs)
            {
                int len = t.first;
                std::uniform_int_distribution<int> u_start(0, data_nb - len);

                for (int i = 0; i < query_nb / 10; i++)
                {
                    int ql = u_start(e);
                    int qr = ql + len - 1;
                    if (ql >= data_nb || qr >= data_nb)
                        throw Exception("Query range out of bound");
                    outfile.write((char *)&ql, sizeof(int));
                    outfile.write((char *)&qr, sizeof(int));
                }
            }
            outfile.close();
        }

        float dis_compute(std::vector<float> &v1, std::vector<float> &v2)
        {
            hnswlib::DISTFUNC<float> fstdistfunc_ = space->get_dist_func();
            float dis = fstdistfunc_((char *)v1.data(), (char *)v2.data(), space->get_dist_func_param());
            return dis;
        }

        void GenerateGroundtruth(std::string saveprefix, DataLoader &storage)
        {
            space = new hnswlib::L2Space(storage.Dim);
            for (auto t : storage.query_range)
            {
                int suffix = t.first;
                std::string savepath = saveprefix + std::to_string(suffix) + ".bin";
                // CheckPath(savepath);
                std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);
                std::cout << "generating for " << t.first << std::endl;
                for (int i = 0; i < query_nb; i++)
                {
                    auto rp = t.second[i];
                    int ql = rp.first, qr = rp.second;
                    std::priority_queue<std::pair<float, int>> ans;
                    for (int j = ql; j <= qr; j++)
                    {
                        float dis = dis_compute(storage.query_points[i], storage.data_points[j]);
                        ans.emplace(dis, j);
                        if (ans.size() > storage.query_K)
                            ans.pop();
                    }
                    while (ans.size())
                    {
                        auto id = ans.top().second;
                        ans.pop();
                        outfile.write((char *)&id, sizeof(int));
                    }
                }
                outfile.close();
            }
        }
    };

    class TreeNode
    {
    public:
        int node_id;
        int lbound, rbound;
        int depth;
        std::vector<TreeNode *> childs;
        TreeNode(int l, int r, int d) : lbound(l), rbound(r), depth(d) {}
    };

    class SegmentTree
    {
    public:
        int ways_ = 2;
        TreeNode *root{nullptr};
        int max_depth{-1};
        std::vector<TreeNode *> treenodes;

        SegmentTree(int data_nb)
        {
            root = new TreeNode(0, data_nb - 1, 0);
        }

        void BuildTree(TreeNode *u)
        {
            if (u == nullptr)
                throw Exception("Tree node is a nullptr");
            treenodes.emplace_back(u);
            max_depth = std::max(max_depth, u->depth);
            int L = u->lbound, R = u->rbound;
            size_t Len = R - L + 1;
            if (L == R)
                return;
            int gap = (R - L + 1) / ways_;
            int res = (R - L + 1) % ways_;

            for (int l = L; l <= R;)
            {
                int r = l + gap - 1;
                if (res > 0)
                {
                    r++;
                    res--;
                }
                r = std::min(r, R);
                TreeNode *childnode = new TreeNode(l, r, u->depth + 1);
                u->childs.emplace_back(childnode);
                BuildTree(childnode);
                l = r + 1;
            }
        }

        std::vector<TreeNode *> range_filter(TreeNode *u, int ql, int qr)
        {
            if (u->lbound >= ql && u->rbound <= qr)
                return {u};
            std::vector<TreeNode *> res;
            if (u->lbound > qr)
                return res;
            if (u->rbound < ql)
                return res;
            for (auto child : u->childs)
            {
                auto t = range_filter(child, ql, qr);
                while (t.size())
                {
                    res.emplace_back(t.back());
                    t.pop_back();
                }
            }
            return res;
        }
    };
}

namespace TimeWindowIndex
{
    class Controller
    {
        public:
            std::vector<TimeDataType> time_data_list;
            int Dim;
            std::ofstream logfile;
            std::vector<TimeDataType> query_list;
            std::vector<VectorDataType> vector_list;
        Controller(std::string outfile){
            logfile.open(outfile);
            if(!logfile.is_open()){
                throw Exception("cannot open " + outfile);
            }
        }
        void LogInsertTime(std::string s){
            logfile << "Insert time is " << s << "ms"<< std::endl;
        }
        void LogAvgTime(std::string s){
            logfile << "Average time is " << s << "ms" << std::endl;
        }
        void LogRemoveTime(std::string s){
            logfile << "Remove time is " << s << "ms" << std::endl;
        }
        void LoadTimeData(std::string vectorfilename, std::string timefilename){
            //ReadVectorData
            ReadVectorData(vectorfilename, vector_list);
            Dim = vector_list[0].Dimension();
            //ReadTimeData
            std::ifstream timefile(timefilename, std::ios::in | std::ios::binary);
            if (!timefile.is_open())
                throw Exception("cannot open " + timefilename);
            
            std::string line;
            std::getline(timefile, line);
            int totalData;
            std::stringstream(line) >> totalData;
            int i = 0;
            while (std::getline(timefile, line)) {
                std::stringstream ss(line);
                int index;
                double tst, ten;
                
                // 读取每行的数据：编号、tst和ten
                char comma;
                if (ss >> index >> tst >> ten) {
                    TimeDataType t(index, tst, ten, vector_list[i]);
                    time_data_list.push_back(t);
                } else {
                    std::cerr << "无法解析数据行: " << line << std::endl;
                }
                i++;
            }

        }
        void GenerateQuery(int queryrange){
            std::mt19937 rng(std::random_device{}());
            query_list.clear();
            

            std::vector<TimeDataType> sample;
            int left_bound = 8640 * (queryrange / 10 - 2);
            int right_bound = left_bound + 8640 * 2;
            int lifetime = 8640 * 2;

            std::sample(time_data_list.begin(), time_data_list.end(), std::back_inserter(sample), 20, std::mt19937(std::random_device{}()));
            for(int i = 0; i < 20; i++){
                std::uniform_int_distribution<int> dist(left_bound, left_bound + lifetime * 2);
                double l = left_bound;
                double r = l + lifetime;
                sample[i].tst = l;
                sample[i].ten = r;
                query_list.emplace_back(sample[i]);
            }
            
            sort(query_list.begin(), query_list.end(), [](const auto &q1, const auto &q2){
                return q1.ten < q2.ten;
            });
        }


        void DumpQuery(std::string filename){
            int32_t query_nb = query_list.size();
            int32_t dim = query_list[0].embedding.Dimension() + 2;
            std::ofstream fout(filename, std::ios::binary);
            if(!fout.is_open()){
                throw Exception("cannot open " + filename);
            }
            assert(Dim == dim - 2);
            fout.write(reinterpret_cast<const char*>(&query_nb), sizeof(int32_t));
            fout.write(reinterpret_cast<const char *>(&dim), sizeof(int32_t));
            for(const auto& q: query_list){
                float tst = q.tst;
                float ten = q.ten;
                fout.write(reinterpret_cast<const char*>(q.embedding.data.data()), (dim - 2) * sizeof(float));
                fout.write(reinterpret_cast<const char *>(&tst), sizeof(float));
                fout.write(reinterpret_cast<const char *>(&ten), sizeof(float));
            }
            fout.close();
        }

        void PrintQuery(){
            int i = -1;
            for(auto r: query_list){
                i++;
                std::cout << "Query " << i << ": " << std::endl;
                std::cout << "Embedding: " << r.embedding.to_string() << std::endl;
                std::cout << "Query Range: [" << r.tst << ", " << r.ten << "]" << std::endl;
            }
        }
        void LoadQuery(std::string filename){
            query_list.clear();
            int32_t query_nb;
            int32_t dim;
            std::ifstream fin(filename, std::ios::binary);
            if(!fin.is_open()){
                throw Exception("cannot open " + filename);
            }
            fin.read(reinterpret_cast<char*>(&query_nb), sizeof(int32_t));
            fin.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
            for(int i = 0; i < query_nb; i++){
                float tst, ten;
                std::vector<float> v(dim - 2);
                fin.read(reinterpret_cast<char *>(v.data()), (dim - 2) * sizeof(float));
                fin.read(reinterpret_cast<char *>(&tst), sizeof(float));
                fin.read(reinterpret_cast<char *>(&ten), sizeof(float));
                VectorDataType vt(dim - 2, -1, v);
                TimeDataType t(-1, tst, ten, vt);
                query_list.push_back(t);
            }
            fin.close();
            // PrintQuery();

        }

        static bool CompareTst(const TimeDataType &t1, const TimeDataType &t2){
            if(t1.tst != t2.tst)
                return t1.tst < t2.tst;
            else
                return t1.vid < t2.vid;
        }
        static bool CompareTen(const TimeDataType &t1, const TimeDataType &t2){
            if(t1.ten != t2.ten)
                return t1.ten < t2.ten;
            else
                return t1.vid < t2.vid; 
        }
        std::queue<std::pair<int, std::pair<double, VectorDataType>>> OperationStream() {
            //1 stands for insert //2 stands for remove
            std::vector<TimeDataType> insert_list = time_data_list;
            std::vector<TimeDataType> remove_list = time_data_list;

            std::sort(insert_list.begin(), insert_list.end(), CompareTst);
            std::sort(remove_list.begin(), remove_list.end(), CompareTen);

            std::queue<std::pair<int, std::pair<double, VectorDataType>>> operations;

            size_t i = 0;
            size_t j = 0;
            size_t n = insert_list.size();
            size_t m = remove_list.size();

            while (i < n && j < m) {
                double tst = insert_list[i].tst;
                double ten = remove_list[j].ten;

                if (tst <= ten) {
                    // 插入事件
                    operations.push({1, {tst, insert_list[i].embedding}});
                    i++;
                } else {
                    // 删除事件
                    operations.push({2, {ten, remove_list[j].embedding}});
                    j++;
                }
            }

            // 剩余插入事件
            while (i < n) {
                operations.push({1, {insert_list[i].tst, insert_list[i].embedding}});
                i++;
            }

            // 剩余删除事件
            while (j < m) {
                operations.push({2, {remove_list[j].ten, remove_list[j].embedding}});
                j++;
            }

            return operations;
        }


    };

}