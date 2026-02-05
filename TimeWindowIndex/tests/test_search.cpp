#include "index.h"
std::unordered_map<std::string, std::string> paths;

const int query_K = 10;
int M;
int ef_construction;
int Dim;
int percent = 100;
std::vector<::iRangeGraph::DataLoader *> storages;


size_t size_of_pair() {
    return sizeof(std::pair<float, int>);
}

size_t size_of_vector_of_pair(const std::vector<std::pair<float, int>>& vec) {
    return sizeof(vec) + vec.size() * size_of_pair();
}

size_t size_of_vector_of_vector_of_pair(const std::vector<std::vector<std::pair<float, int>>>& vec) {
    size_t total_size = sizeof(vec);  // vector 的结构
    for (const auto& v : vec) {
        total_size += size_of_vector_of_pair(v);
    }
    return total_size;
}

size_t size_of_edges(const std::vector<std::vector<std::vector<std::pair<float, int>>>>& vec) {
    size_t total_size = sizeof(vec);  // vector 的结构
    for (const auto& v1 : vec) {
        total_size += size_of_vector_of_vector_of_pair(v1);
    }
    return total_size;
}

double bytes_to_MB(size_t bytes) {
    return static_cast<double>(bytes) / (1024 * 1024);
}

int index_size = 0;
void CalculateLiveIndexSize(TimeWindowIndex::LiveIndex<float> *lindex){
    std::unordered_set<int> globalids;
    std::vector<std::shared_ptr<TimeWindowIndex::iRangeGraph<float>>> irgs;
    for(auto pair: lindex->baseindex){
        std::shared_ptr<TimeWindowIndex::iRangeGraph<float>> irg = pair.second;
        irgs.push_back(irg);
    }
    int sum = 0;
    for(int i = 0; i < irgs.size(); i++){
        int data_nb = irgs[i]->storage->data_nb;
        storages.push_back(new ::iRangeGraph::DataLoader());
        storages[i]->data_points = irgs[i]->storage->data_points;
        storages[i]->vid_hash = irgs[i]->storage->vid_hash;

        storages[i]->t_hash = irgs[i]->storage->t_hash;
        storages[i]->fusion_hash = irgs[i]->storage->fusion_hash;
        storages[i]->Dim = irgs[i]->storage->Dim;
        storages[i]->data_nb = irgs[i]->storage->data_nb;
        sum += storages[i]->data_nb;
        std::shared_ptr<TimeWindowIndex::iRangeGraph<float>> irg = std::make_shared<TimeWindowIndex::iRangeGraph<float>>(storages[i], M, ef_construction);
        lindex->baseindex[irgs[i]->builder->edges.front().size()] = irg;
        index_size += size_of_edges(irg->builder->edges);
    }
}

void CalculateDeadIndexSize(TimeWindowIndex::DeadIndex<float> *dindex){
    for(auto i: dindex->baseindex){
        CalculateLiveIndexSize(i->index);
    }
}

int main(int argc, char **argv){
    

    for(int i = 0; i < argc; i++){
        std::string arg = argv[i];
        if(arg == "--vector_path")
            paths["vector_path"] = argv[i + 1];
        if(arg == "--time_path")
            paths["time_path"] = argv[i + 1];
        if(arg == "--M")
            M = std::stoi(argv[i + 1]);
        if(arg == "--ef_construction")
            ef_construction = std::stoi(argv[i + 1]);
        if(arg == "--Dim")
            Dim = std::stoi(argv[i + 1]);
        if(arg == "--log_path")
            paths["log_path"] = argv[i + 1];
        if(arg == "--res_path")
            paths["res_path"] = argv[i + 1];
        if(arg == "--query_path")
            paths["query_path"] = argv[i + 1];
        if(arg == "--percent")
            percent = std::stoi(argv[i + 1]);
        }
    if(paths["vector_path"] == ""){
        throw Exception("Null vector path");
    }
    if(paths["time_path"] == ""){
        throw Exception("Null time path");
    }
    if(M <= 0 || ef_construction <= 0){
        throw Exception("Wrong Parameters");
    }
    TimeWindowIndex::Controller * c = new TimeWindowIndex::Controller(paths["log_path"]);
    // std::unordered_map<int, double> tst_hash;
    std::vector<double> tst_hash;
    std::unordered_map<int, double> ten_hash;

    c->LoadTimeData(paths["vector_path"], paths["time_path"]);
    std::cout << "Load Time Data Done" << std::endl;
    if(Dim != c->Dim){
        throw Exception("Wrong Input Dimensionality, and the correct one is " + std::to_string(128));
    }
    std::queue<std::pair<int, std::pair<double, VectorDataType>>> operations = c->OperationStream();
    std::cout << "Get operations done" << std::endl;
    TimeWindowIndex::LiveIndex<float> *lindex = new TimeWindowIndex::LiveIndex<float>(M, ef_construction, Dim);
    TimeWindowIndex::DeadIndex<float> *dindex = new TimeWindowIndex::DeadIndex<float>(M, ef_construction, -1, Dim, false);
    float inserttime = 0.0;
    float removetime = 0.0;
    int insertnb = 0;
    int removenb = 0;
    c->LoadQuery(paths["query_path"]);
    std::cout << "Load Query Done" << std::endl;
    int op_nb = operations.size() * percent / 100;
    while(!operations.empty()){
        int opcode = operations.front().first;
        if(opcode == 1){
            std::vector<std::pair<double, VectorDataType>> insert_list;
            int t = operations.front().second.first;
            insert_list.emplace_back(operations.front().second);
            insertnb++;
            tst_hash.push_back(t);
            timeval t1, t2;
            gettimeofday(&t1, NULL);
            lindex->Insert(insert_list);
            gettimeofday(&t2, NULL);
            auto duration = GetTime(t1, t2);
            inserttime += duration;
            operations.pop();
        }else{
            std::vector<std::pair<double, VectorDataType>> remove_list;
            int t = operations.front().second.first;
            remove_list.emplace_back(operations.front().second);
            removenb++;
            timeval t1, t2;
            std::vector<double> tst_list;
            ten_hash.insert({operations.front().second.second.vid, t});
            tst_list.push_back(tst_hash[operations.front().second.second.vid]);
            gettimeofday(&t1, NULL);
            lindex->Remove(remove_list);
            dindex->Insert(remove_list, tst_list);
            gettimeofday(&t2, NULL);
            auto duration = GetTime(t1, t2);
            removetime += duration;
            operations.pop();
        }
        if(insertnb + removenb > op_nb){
            break;
        }
    }
    c->logfile << "Insert " << insertnb << " datas:" << std::endl;
    c->LogAvgTime(std::to_string(inserttime / insertnb * 1000));
    c->logfile << "Remove " << removenb << " datas:" << std::endl;
    c->LogAvgTime(std::to_string(removetime / removenb * 1000));
    c->logfile << "Avg update time is " << (inserttime + removetime) / (insertnb + removenb) * 1000 << std::endl;
    c->logfile << "Update Throughput is " << (insertnb + removenb) /(inserttime + removetime)  << std::endl;

    c->logfile << "Avg time for single operation is " << (removetime + inserttime) * 1000 / (insertnb + removenb);
    std::cout << "Avg time for single operation is " << (removetime + inserttime) * 1000 / (insertnb + removenb) << std::endl;
    std::vector<int> efs = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
    TimeWindowIndex::Controller *res_logger = new TimeWindowIndex::Controller(paths["res_path"]);

    CalculateLiveIndexSize(lindex);
    CalculateDeadIndexSize(dindex);

    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "Index Size is " << bytes_to_MB(index_size) << "MB" << std::endl; 
    std::cout << "-----------------------------------------------" << std::endl;
    for(int ef: efs){
        int i = -1;
        float searchtime = 0.0;
        res_logger->logfile << "EF = " << ef << std::endl;
        {for(TimeDataType t: c->query_list){
            i++;
            timeval t1, t2;
            double qtst = t.tst;
            double qten = t.ten;
            std::priority_queue<std::pair<float, int>> final_res;

            gettimeofday(&t1, NULL);
            std::priority_queue<std::pair<float, int>> liveres = lindex->Search(ef, qten, t.embedding, query_K);
            std::priority_queue<std::pair<float, int>> deadres = dindex->Search(ef, qtst, qten, t.embedding, query_K);
            std::unordered_set<int> res_record;
            while(!liveres.empty()){
                if(ten_hash[liveres.top().second] >= qtst && tst_hash[liveres.top().second] <= qten)
                {                
                    final_res.push(liveres.top());
                    res_record.insert(liveres.top().second);
                }
                liveres.pop();
            }
            while(!deadres.empty()){
                if(tst_hash[deadres.top().second] <= qten && res_record.find(deadres.top().second) == res_record.end() && ten_hash[deadres.top().second] >= qtst){
                    final_res.push(deadres.top());
                }
                deadres.pop();
            }


            int j = 0;

            gettimeofday(&t2, NULL);
            double duration = GetTime(t1, t2);
            searchtime += duration * 1000;
            while(final_res.size() > query_K){
                final_res.pop();
            }

            while(!final_res.empty()){
                res_logger->logfile << "Ans " << j << ": vid: " << final_res.top().second << " dis: " << std::sqrt(final_res.top().first) << std::endl;
                final_res.pop();
                j++;
            }
            res_logger->logfile << "Query " << i  << " costs " << duration * 1000 << "ms " << std::endl;
        }
        res_logger->logfile  << "Avg Search Time: " << searchtime / (i + 1) << " ms\nQPS: " << (i + 1) * 1000 / searchtime << std::endl;
        std::cout <<  "EF=" << ef << std::endl << "Avg Search Time: " << searchtime / (i + 1) << " ms\nQPS: " << (i + 1) * 1000 / searchtime << std::endl;}
    }
    return 0;

}