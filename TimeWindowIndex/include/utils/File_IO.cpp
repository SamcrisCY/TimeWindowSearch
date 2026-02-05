/**
	@author:	Yuxiang Zeng
	@email: 	yxzeng@buaa.edu.cn
	@date:		2024.09.28
*/
#include "File_IO.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <set>

// Check the suffix of a specific file
static bool EndsWith(const std::string &str, const std::string &suffix);

// READ from .txt file
static void __ReadVectorDataFromTxt(const std::string &file_name, std::vector <VectorDataType> &data_list);

static void __ReadVectorQueryFromTxt(const std::string &file_name, std::vector <VectorDataType> &query_data_list,
                                     std::vector <size_t> &query_k_list);

static void __ReadHybridVectorQueryFromTxt(const std::string &file_name, std::vector<VectorDataType> &query_data_list,
                                    std::vector<size_t> &query_k_list, std::vector<std::string> &condition_list);

static void __ReadGroundTruthFromTxt(const std::string &file_name, std::vector <std::vector<VidType>> &answer_list);

// DUMP into .txt file
static void __DumpVectorQueryToTxt(const std::string &file_name, const std::vector <VectorDataType> &query_data_list,
                                   const std::vector <size_t> &query_k_list);

static void __DumpVectorDataToTxt(const std::string &file_name, const std::vector <VectorDataType> &data_list);

static void __DumpGroundTruthToTxt(const std::string &file_name, const std::vector <std::vector<VidType>> &answer_list);

static void __ReadScalarDataFromTxt(const std::string &file_name, std::vector<std::shared_ptr<attribute>>& data_list);

// READ from .fvecs, .bvecs, .ivecs file
static void __ReadVectorDataFromFvecs(const std::string &file_name, std::vector <VectorDataType> &data_list);

static void __ReadVectorDataFromBvecs(const std::string &file_name, std::vector <VectorDataType> &data_list);

static void __ReadVectorQueryFromFvecs(const std::string &file_name, std::vector <VectorDataType> &query_data_list,
                                       std::vector <size_t> &query_k_list);

static void __ReadVectorQueryFromBvecs(const std::string &file_name, std::vector <VectorDataType> &query_data_list,
                                       std::vector <size_t> &query_k_list);

static void __ReadGroundTruthFromIvecs(const std::string &file_name, std::vector <std::vector<VidType>> &answer_list);

static void __DumpVectorQueryToFvecs(const std::string &file_name, const std::vector <VectorDataType> &query_data_list,
                                     const std::vector <size_t> &query_k_list);

static void __DumpVectorDataToFvecs(const std::string &file_name, const std::vector <VectorDataType> &data_list);
static void __DumpVectorDataToBin(const std::string &file_name, const std::vector <VectorDataType> &data_list);
static void
__DumpGroundTruthToIvecs(const std::string &file_name, const std::vector <std::vector<VidType>> &answer_list);


/// both vid and data are stored for each vector data
static void __ReadVectorDataFromFivecs(const std::string &file_name, std::vector <VectorDataType> &data_list);

static void __DumpVectorDataToFivecs(const std::string &file_name, const std::vector <VectorDataType> &data_list);

//READ from .fbin .bin
static void __ReadVectorDataFromFbin(const std::string &file_name, std::vector <VectorDataType> &data_list);

static void __ReadVectorQueryFromFbin(const std::string &file_name, std::vector <VectorDataType> &query_data_list,
                                      std::vector <size_t> &query_k_list);

static void __ReadGroundTruthFromBin(const std::string &file_name, std::vector <std::vector<VidType>> &answer_list);

static void __DumpGroundTruthToBin(const std::string &file_name, std::vector <VectorDataType> &data_list);


static bool EndsWith(const std::string &str, const std::string &suffix) {
    if (suffix.size() > str.size()) {
        return false;
    }
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void ReadVectorData(const std::string &file_name, std::vector <VectorDataType> &data_list) {
    if (EndsWith(file_name, ".txt")) {
        __ReadVectorDataFromTxt(file_name, data_list);
    } else if (EndsWith(file_name, ".fvecs")) {
        __ReadVectorDataFromFvecs(file_name, data_list);
    } else if (EndsWith(file_name, ".fivecs")) {
        __ReadVectorDataFromFivecs(file_name, data_list);
    } else if (EndsWith(file_name, ".bvecs")) {
        __ReadVectorDataFromBvecs(file_name, data_list);
    } else {
        __ReadVectorDataFromFbin(file_name, data_list);
    }
}

void ReadScalarData(const std::string& file_name, std::vector<std::shared_ptr<attribute>>& data_list) {
    if (EndsWith(file_name, ".txt")) {
        __ReadScalarDataFromTxt(file_name, data_list);
    }
}

void DumpVectorData(const std::string &file_name, const std::vector <VectorDataType> &data_list) {
    if (EndsWith(file_name, ".txt")) {
        __DumpVectorDataToTxt(file_name, data_list);
    } else if (EndsWith(file_name, ".fvecs")) {
        __DumpVectorDataToFvecs(file_name, data_list);
    } else if (EndsWith(file_name, ".fivecs")) {
        __DumpVectorDataToFivecs(file_name, data_list);
    } else if (EndsWith(file_name, ".bin")){
        __DumpVectorDataToBin(file_name, data_list);
    } else{
        std::cerr << "Unknown file type: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void ReadVectorQuery(const std::string &file_name, std::vector <VectorDataType> &query_data_list,
                     std::vector <size_t> &query_k_list) {
    if (EndsWith(file_name, ".txt")) {
        __ReadVectorQueryFromTxt(file_name, query_data_list, query_k_list);
    } else if (EndsWith(file_name, ".fvecs")) {
        __ReadVectorQueryFromFvecs(file_name, query_data_list, query_k_list);
    } else if (EndsWith(file_name, ".bvecs")) {
        __ReadVectorQueryFromBvecs(file_name, query_data_list, query_k_list);
    } else if (EndsWith(file_name, ".fbin") || EndsWith(file_name, ".bin")) {
        __ReadVectorQueryFromFbin(file_name, query_data_list, query_k_list);
    } else {
        std::cerr << "Unknown file type: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void ReadHybridVectorQuery(const std::string &file_name, std::vector<VectorDataType> &query_data_list,
                    std::vector<size_t> &query_k_list, std::vector<std::string> &condition_list) {
     if (EndsWith(file_name, ".txt")) {
        __ReadHybridVectorQueryFromTxt(file_name, query_data_list, query_k_list, condition_list);
    }
}

void DumpVectorQuery(const std::string &file_name, const std::vector <VectorDataType> &query_data_list,
                     const std::vector <size_t> &query_k_list) {
    if (EndsWith(file_name, ".txt")) {
        __DumpVectorQueryToTxt(file_name, query_data_list, query_k_list);
    } else if (EndsWith(file_name, ".fvecs")) {
        __DumpVectorQueryToFvecs(file_name, query_data_list, query_k_list);
    } else {
        std::cerr << "Unknown file type: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void ReadGroundTruth(const std::string &file_name, std::vector <std::vector<VidType>> &answer_list) {
    if (EndsWith(file_name, ".txt")) {
        __ReadGroundTruthFromTxt(file_name, answer_list);
    } else if (EndsWith(file_name, ".ivecs")) {
        __ReadGroundTruthFromIvecs(file_name, answer_list);
    } else if (EndsWith(file_name, ".bin")) {
        __ReadGroundTruthFromBin(file_name, answer_list);
    } else {
        std::cerr << "Unknown file type: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void DumpGroundTruth(const std::string &file_name, const std::vector <std::vector<VidType>> &answer_list) {
    if (EndsWith(file_name, ".txt")) {
        __DumpGroundTruthToTxt(file_name, answer_list);
    } else if (EndsWith(file_name, ".ivecs")) {
        __DumpGroundTruthToIvecs(file_name, answer_list);
    } else {
        std::cerr << "Unknown file type: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void ReadSiloIPaddr(const std::string &file_name, std::vector <std::string> &silo_ipaddr_list) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading data silos' IP addresses: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    size_t n;
    if (!(file >> n)) {
        std::cerr << "Failed to read n from file: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    silo_ipaddr_list.resize(n);
    for (size_t i = 0; i < n; ++i) {
        if (!(file >> silo_ipaddr_list[i])) {
            std::cerr << "Failed to read data silos' IP addresses from line " << (i + 2) << " in file: " << file_name
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    file.close();
}

void DumpSiloIPaddr(const std::string &file_name, const std::vector <std::string> &silo_ipaddr_list) {
    std::ofstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading data silos' IP addresses: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    size_t n = silo_ipaddr_list.size();

    file << n << "\n";
    for (auto ipaddr: silo_ipaddr_list) {
        file << ipaddr << "\n";
    }
    file.close();
}

void SplitVectorData(const std::string &file_name, const int silo_num, const std::string &des_file_path,
                     const std::vector <VectorDataType> data_list) {
    std::string file_path = des_file_path;
    size_t last_slash_pos = file_name.find_last_of('/');

    if (file_path.empty()) {
        if (last_slash_pos == std::string::npos) {
            file_path = "./";
        } else {
            file_path = file_name.substr(0, last_slash_pos);
        }
    }

    std::string file_name_token;
    std::string file_name_prefix, file_name_suffix;

    {
        if (last_slash_pos == std::string::npos) {
            file_name_token = file_name;
        } else {
            file_name_token = file_name.substr(last_slash_pos + 1);
        }

        std::cout << "file_path: " << file_path << ", file_name = " << file_name_token << std::endl;
        size_t last_dot_pos = file_name_token.find_last_of('.');
        if (last_dot_pos == std::string::npos) {
            file_name_prefix = file_name_token;
            file_name_suffix = "";
        } else {
            file_name_prefix = file_name_token.substr(0, last_dot_pos);
            file_name_suffix = file_name_token.substr(last_dot_pos);
        }
        std::cout << "file_name_prefix: " << file_name_prefix << ", file_name_suffix = " << file_name_suffix
                  << std::endl;
    }

    if (silo_num <= 0) {
        std::string error_message;
        error_message = std::string("Invalid silo_num = ") + std::to_string(silo_num) + std::string("when splitting");
        throw std::invalid_argument(error_message);
    }

    size_t data_size = data_list.size();
    size_t silo_data_size = data_size / silo_num;
    size_t cur_silo_data_size, idx = 0;

    for (int i = 1; i <= silo_num; ++i) {
        cur_silo_data_size = silo_data_size;
        if (i <= data_size % silo_num) {
            ++cur_silo_data_size;
        }

        std::vector <VectorDataType> silo_data_list(data_list.begin() + idx,
                                                    data_list.begin() + idx + cur_silo_data_size);
        idx += cur_silo_data_size;

        std::string des_file_name;
//        if (file_name_suffix == ".fvecs") {
        des_file_name = file_name_prefix + std::string("_") + std::to_string(i - 1) + std::string(".fivecs");
//        } else {
//            des_file_name = file_name_prefix + std::string("_") + std::to_string(i-1) + file_name_suffix;
//        }

        des_file_name = file_path + std::string("/") + des_file_name;
        DumpVectorData(des_file_name, silo_data_list);

        std::cout << "Dump Silo #" << i - 1 << "'s " << std::to_string(cur_silo_data_size) << " vector data into "
                  << des_file_name << std::endl;
    }
}

void SplitVectorDataForDigData(const std::string &file_name, const int silo_num, const std::string &des_file_path) {
    std::string file_path = des_file_path;
    size_t last_slash_pos = file_name.find_last_of('/');

    if (file_path.empty()) {
        if (last_slash_pos == std::string::npos) {
            file_path = "./";
        } else {
            file_path = file_name.substr(0, last_slash_pos);
        }
    }

    std::string file_name_token;
    std::string file_name_prefix, file_name_suffix;

    {
        if (last_slash_pos == std::string::npos) {
            file_name_token = file_name;
        } else {
            file_name_token = file_name.substr(last_slash_pos + 1);
        }

        std::cout << "file_path: " << file_path << ", file_name = " << file_name_token << std::endl;
        size_t last_dot_pos = file_name_token.find_last_of('.');
        if (last_dot_pos == std::string::npos) {
            file_name_prefix = file_name_token;
            file_name_suffix = "";
        } else {
            file_name_prefix = file_name_token.substr(0, last_dot_pos);
            file_name_suffix = file_name_token.substr(last_dot_pos);
        }
        std::cout << "file_name_prefix: " << file_name_prefix << ", file_name_suffix = " << file_name_suffix
                  << std::endl;
    }

    if (silo_num <= 0) {
        std::string error_message;
        error_message = std::string("Invalid silo_num = ") + std::to_string(silo_num) + std::string("when splitting");
        throw std::invalid_argument(error_message);
    }

    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading vector data: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }
    VidType vid = 0;
    int32_t nvecs, dim;
    file.read(reinterpret_cast<char *>(&nvecs), sizeof(int32_t));
    file.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
    std::cout << "nvecs: " << nvecs << ", dim: " << dim << std::endl;
    size_t silo_data_size = nvecs / silo_num;
    std::cout << "silo_data_size: " << silo_data_size << std::endl;
    std::vector <VectorDataType> silo_data_list;
    for (int i = 1; i <= silo_num; ++i) {
        size_t cur_silo_data_size = silo_data_size;
        if (i <= nvecs % silo_num) {
            ++cur_silo_data_size;
        }
        std::vector<float> arr((cur_silo_data_size * dim));
        std::cout << "start" << std::endl;
        file.read(reinterpret_cast<char *>(arr.data()), cur_silo_data_size * dim * sizeof(float));
        if (!file) {
            throw std::runtime_error("Error reading file");
        }
        for (int i = 0; i < cur_silo_data_size; ++i) {
            VectorDataType vector_data(dim, vid++);
            for (int j = 0; j < dim; ++j) {
                vector_data.data[j] = arr[i * dim + j];
            }
            silo_data_list.emplace_back(vector_data);
        }
        std::cout << "finish" << std::endl;
        std::string des_file_name;
        des_file_name = file_name_prefix + std::string("_") + std::to_string(i - 1) + std::string(".fivecs");
        des_file_name = file_path + std::string("/") + des_file_name;
        std::ofstream file_out(des_file_name, std::ios::binary);
        if (!file_out.is_open()) {
            std::cerr << "Failed to open file for writing vector data: " << des_file_name << std::endl;
            std::exit(EXIT_FAILURE);
        }
        int32_t sub_nvecs = static_cast<int32_t>(silo_data_list.size());
        file_out.write(reinterpret_cast<const char *>(&sub_nvecs), sizeof(int32_t));
        file_out.write(reinterpret_cast<const char *>(&dim), sizeof(int32_t));
        for (const auto &vector_data: silo_data_list) {
            file_out.write(reinterpret_cast<const char *>(&vector_data.vid), sizeof(VidType));
            file_out.write(reinterpret_cast<const char *>(vector_data.data.data()), dim * sizeof(VectorDimensionType));
        }
        file_out.close();
        silo_data_list.clear();
    }
    file.close();
}

void EvaluateAnswer(const std::string &answer_filename, const std::string &truth_filename) {
    std::vector <std::vector<VidType>> answer_list;
    std::vector <std::vector<VidType>> truth_list;

    ReadGroundTruth(answer_filename, answer_list);
    ReadGroundTruth(truth_filename, truth_list);

    size_t query_num = answer_list.size();

    if (query_num != truth_list.size()) {
        std::cerr << "Evaluation answer with sizes different from ground truth: " << answer_list.size() << " vs "
                  << truth_list.size() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    double avg_recall = 0.0;

    for (size_t i = 0; i < query_num; ++i) {
        std::set <VidType> truth_set(truth_list[i].begin(), truth_list[i].end());
        double recall = 0.0;

        for (VidType vid: answer_list[i]) {
            if (truth_set.count(vid) > 0) {
                recall += 1.0;
            }
        }
        if (truth_list[i].size() > 0)
            recall /= truth_list[i].size();

        avg_recall += recall;

        std::cout << "Query #" << i + 1 << ": recall = " << std::fixed << std::setprecision(6) << recall << std::endl;
    }

    if (query_num > 0)
        avg_recall /= query_num;

    std::cout << query_num << " queries: recall = " << std::fixed << std::setprecision(6) << avg_recall << " per query"
              << std::endl;
}

void ReadTimeData(const std::string& file_name, std::vector<TimeDataType>& time_list){
    std::ifstream ifs(file_name, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open file for reading: " + file_name);

    size_t size = 0;
    ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
    // std::cout << "size is " << size;
    time_list.resize(size);
    ifs.read(reinterpret_cast<char*>(time_list.data()), sizeof(TimeDataType) * size);

}
void DumpTimeData(const std::string& file_name, const std::vector<TimeDataType>& time_list){
    std::ofstream ofs(file_name, std::ios::binary);
    if (!ofs) throw std::runtime_error("Failed to open file for writing: " + file_name);

    size_t size = time_list.size();
    ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));  // 写入数量
    ofs.write(reinterpret_cast<const char*>(time_list.data()), sizeof(TimeDataType) * size);
}

static void __ReadScalarDataFromTxt(const std::string &file_name, std::vector<std::shared_ptr<attribute>>& data_list) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading scalar data: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    int n, m; //data number, attribute number
    if (!(iss >> n >> m)) {
        std::cerr << "Failed to read data number and attribute number from file: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::string name, type;
    std::getline(file, line);
    std::istringstream isy(line);
    for(int i = 0;i < m;i++) {
        isy >> name >> type;  
        std::shared_ptr<attribute> item;   
        if(type == "string") {
            item = std::make_shared<StringType>(name, n);  
        } else if(type == "int") {
            item = std::make_shared<IntType>(name, n);   
        } else {
            item = std::make_shared<FloatType>(name, n);   
        }
        data_list.emplace_back(item);
    }
    for(int i = 0;i < n;i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        for(int j = 0;j < m;j++) {
            if(data_list[j]->getType() == STRING) {
                std::string now;
                iss >> now;
                dynamic_cast<StringType*>(data_list[j].get())->insert(i, now);
            } else if(data_list[j]->getType() == INT) {
                int now;
                iss >> now;
                dynamic_cast<IntType*>(data_list[j].get())->insert(i, now);
            } else {
                float now;
                iss >> now;
                dynamic_cast<FloatType*>(data_list[j].get())->insert(i, now);
            }
        }
    }
    file.close();
    for(int i = 0;i < m;i++) {
        if(data_list[i]->getType() == STRING) {
            dynamic_cast<StringType*>(data_list[i].get())->Discretization();
        }
    }
    std::cout << "Read data: size = " << n << ", scalar data' number = " << m << std::endl;
    return;
}

static void __ReadVectorDataFromTxt(const std::string &file_name, std::vector <VectorDataType> &data_list) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading vector data: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    size_t n, dim;
    if (!(iss >> n >> dim)) {
        std::cerr << "Failed to read n and dim from file: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    data_list.clear();
    data_list.resize(n, VectorDataType(dim));

    for (int i = 0; i < n; ++i) {
        std::getline(file, line);
        std::istringstream line_iss(line);
        if (!(line_iss >> data_list[i].vid)) {
            std::cerr << "Failed to read vector_data.vid from line " << (i + 2) << " in file: " << file_name
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        data_list[i].data.reserve(dim);
        data_list[i].data.resize(dim);
        for (int j = 0; j < dim; ++j) {
            if (!(line_iss >> data_list[i].data[j])) {
                std::cerr << "Failed to read vector_data.data from line " << (i + 2) << " in file: " << file_name
                          << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }

    file.close();
}

static void __DumpVectorDataToTxt(const std::string &file_name, const std::vector <VectorDataType> &data_list) {
    std::ofstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for dumping vector data: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    size_t n = data_list.size(), dim = 1;
    if (n == 0) {
        file << n << " " << dim << "\n";
        file.close();
        return;
    }

    dim = data_list[0].Dimension();
    file << n << " " << dim << "\n";
    for (const auto &vector_data: data_list) {
        file << vector_data.vid;
        for (auto d: vector_data.data) {
            file << " " << d;
        }
        file << "\n";
    }

    file.close();
}

static void __ReadVectorQueryFromTxt(const std::string &file_name, std::vector <VectorDataType> &query_data_list,
                                     std::vector <size_t> &query_k_list) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading vector queries: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    size_t n, dim;
    if (!(iss >> n >> dim)) {
        std::cerr << "Failed to read n and dim from file: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    query_data_list.clear();
    query_data_list.resize(n, VectorDataType(dim));
    query_k_list.clear();
    query_k_list.resize(n);

    for (int i = 0; i < n; ++i) {
        std::getline(file, line);
        std::istringstream line_iss(line);

        query_data_list[i].vid = i;
        query_data_list[i].data.reserve(dim);
        query_data_list[i].data.resize(dim);
        for (int j = 0; j < dim; ++j) {
            if (!(line_iss >> query_data_list[i].data[j])) {
                std::cerr << "Failed to read vector_query.data from line " << (i + 2) << " in file: " << file_name
                          << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        if (!(line_iss >> query_k_list[i])) {
            std::cerr << "Failed to read vector_query.k from line " << (i + 2) << " in file: " << file_name
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (i == 100) break;
    }

    file.close();
}

static void __ReadHybridVectorQueryFromTxt(const std::string &file_name, std::vector<VectorDataType> &query_data_list,
                    std::vector<size_t> &query_k_list, std::vector<std::string> &condition_list) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading vector queries: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    size_t n, dim;
    if (!(iss >> n >> dim)) {
        std::cerr << "Failed to read n and dim from file: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    query_data_list.clear();
    query_data_list.resize(n, VectorDataType(dim));
    query_k_list.clear();
    query_k_list.resize(n);
    condition_list.clear();
    condition_list.resize(n);
    
    for (int i = 0; i < n; ++i) {
        std::getline(file, line);
        std::istringstream line_iss(line);

        query_data_list[i].vid = i;
        query_data_list[i].data.reserve(dim);
        query_data_list[i].data.resize(dim);
        for (int j = 0; j < dim; ++j) {
            if (!(line_iss >> query_data_list[i].data[j])) {
                std::cerr << "Failed to read vector_query.data from line " << (i + 2) << " in file: " << file_name
                          << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        std::string condition = "";
        std::string token;
        if(!(line_iss >> token)) {
            std::cerr << "Failed to read condition from file: " << file_name << std::endl;
            std::exit(EXIT_FAILURE);
        }
        condition.append(token);
        if (line_iss >> token) {
            condition.append(" and ");
            condition.append(token);
        }
        condition_list[i] = condition;
    }
    file.close();
}

static void __DumpVectorQueryToTxt(const std::string &file_name, const std::vector <VectorDataType> &query_data_list,
                                   const std::vector <size_t> &query_k_list) {
    std::ofstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for dumping vector queries: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    size_t n = query_data_list.size(), dim = 1;
    if (n == 0) {
        file << n << " " << dim << "\n";
        file.close();
        return;
    }

    dim = query_data_list[0].Dimension();
    file << n << " " << dim << "\n";
    for (size_t i = 0; i < n; ++i) {
        for (auto d: query_data_list[i].data) {
            file << d << " ";
        }
        file << query_k_list[i] << "\n";
    }

    file.close();
}

static void __ReadGroundTruthFromTxt(const std::string &file_name, std::vector <std::vector<VidType>> &answer_list) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading ground truth: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    size_t n, k;
    if (!(iss >> n)) {
        std::cerr << "Failed to read n from file: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    answer_list.clear();
    for (int i = 0; i < n; ++i) {
        std::getline(file, line);
        std::istringstream line_iss(line);
        if (!(line_iss >> k)) {
            std::cerr << "Failed to read ground truth k from line " << (i + 2) << " in file: " << file_name
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        std::vector <VidType> answer(k);
        answer.reserve(k);
        for (int j = 0; j < k; ++j) {
            if (!(line_iss >> answer[j])) {
                std::cerr << "Failed to read ground truth vid from line " << (i + 2) << " in file: " << file_name
                          << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        answer_list.emplace_back(answer);
    }

    file.close();
}

static void
__DumpGroundTruthToTxt(const std::string &file_name, const std::vector <std::vector<VidType>> &answer_list) {
    std::ofstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for dumping vector queries: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    size_t n = answer_list.size(), k = 0;

    file << n << "\n";
    for (size_t i = 0; i < n; ++i) {
        file << answer_list[i].size();
        for (auto vid: answer_list[i]) {
            file << " " << vid;
        }
        file << "\n";
    }
    file.close();
}

static void __ReadVectorDataFromFvecs(const std::string &file_name, std::vector <VectorDataType> &data_list) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading vector data: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    data_list.clear();
    VidType vid = 0;
    while (file) {
        int32_t dim;
        file.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
        if (file.eof()) break; // End of file reached

        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
        if (!file) {
            throw std::runtime_error("Error reading file");
        }

        VectorDataType vector_data(dim, vid++);
        for (int i = 0; i < dim; ++i)
            vector_data.data[i] = vec[i];
        data_list.emplace_back(vector_data);
    }
    file.close();
}

static void __ReadVectorDataFromFbin(const std::string &file_name, std::vector <VectorDataType> &data_list) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading vector data: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    data_list.clear();
    VidType vid = 0;
    int32_t nvecs, dim;
    file.read(reinterpret_cast<char *>(&nvecs), sizeof(int32_t));
    file.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
    std::vector<float> arr((nvecs * dim));
    file.read(reinterpret_cast<char *>(arr.data()), nvecs * dim * sizeof(float));
    if (!file) {
        throw std::runtime_error("Error reading file");
    }
    for (int i = 0; i < nvecs; ++i) {
        VectorDataType vector_data(dim, vid++);
        for (int j = 0; j < dim; ++j) {
            vector_data.data[j] = arr[i * dim + j];
        }
        data_list.emplace_back(vector_data);
    }
    file.close();
}


static void __ReadVectorDataFromBvecs(const std::string &file_name, std::vector <VectorDataType> &data_list) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading vector data: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    data_list.clear();
    VidType vid = 0;
    while (file) {
        int32_t dim;
        file.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
        if (file.eof()) break; // End of file reached

        std::vector <uint8_t> vec(dim);
        file.read(reinterpret_cast<char *>(vec.data()), dim);
        if (!file) {
            throw std::runtime_error("Error reading file");
        }

        VectorDataType vector_data(dim, vid++);
        for (int i = 0; i < dim; ++i)
            vector_data.data[i] = vec[i];
        data_list.emplace_back(vector_data);
    }
    file.close();
}

static void __ReadVectorDataFromFivecs(const std::string &file_name, std::vector <VectorDataType> &data_list) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading vector data: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    data_list.clear();

    int32_t nvecs, dim;
    file.read(reinterpret_cast<char *>(&nvecs), sizeof(int32_t));
    file.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
    std::cout << "Read data: size = " << nvecs << ", dimension = " << dim << std::endl;

    for (int i = 0; i < nvecs; ++i) {
        VidType vid;
        file.read(reinterpret_cast<char *>(&vid), sizeof(VidType));
        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
        if (!file) {
            throw std::runtime_error("Error reading file");
        }

        VectorDataType vector_data(dim, vid);
        for (int j = 0; j < dim; ++j)
            vector_data.data[j] = vec[j];
        data_list.emplace_back(vector_data);
    }
    file.close();
}


static void __ReadVectorQueryFromFvecs(const std::string &file_name, std::vector <VectorDataType> &query_data_list,
                                       std::vector <size_t> &query_k_list) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading vector query: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    query_data_list.clear();
    VidType vid = 0;
    while (file) {
        int32_t dim;
        file.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
        if (file.eof()) break; // End of file reached

        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
        if (!file) {
            throw std::runtime_error("Error reading file");
        }

        VectorDataType vector_data(dim, vid++);
        for (int i = 0; i < dim; ++i)
            vector_data.data[i] = vec[i];
        query_data_list.emplace_back(vector_data);
        if (vid == 100) break;
    }
    file.close();

    query_k_list.resize(query_data_list.size());
    if (file_name.find("bigann") != std::string::npos) {
        std::fill(query_k_list.begin(), query_k_list.end(), (size_t) 1000);
    } else {
        std::fill(query_k_list.begin(), query_k_list.end(), (size_t) 100);
    }
}

static void __ReadVectorQueryFromFbin(const std::string &file_name, std::vector <VectorDataType> &query_data_list,
                                      std::vector <size_t> &query_k_list) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading vector query: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    query_data_list.clear();
    VidType vid = 0;
    int32_t nvecs, dim;
    file.read(reinterpret_cast<char *>(&nvecs), sizeof(int32_t));
    file.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
    std::vector<float> arr(nvecs * dim);
    file.read(reinterpret_cast<char *>(arr.data()), nvecs * dim * sizeof(float));
    if (!file) {
        throw std::runtime_error("Error reading file");
    }
    for (int i = 0; i < nvecs; ++i) {
        VectorDataType vector_data(dim, vid++);
        for (int j = 0; j < dim; ++j) {
            vector_data.data[j] = arr[i * dim + j];
        }
        query_data_list.emplace_back(vector_data);
        if (i == 100) break;
    }
    file.close();

    query_k_list.resize(query_data_list.size());
    if (file_name.find("bigann") != std::string::npos) {
        std::fill(query_k_list.begin(), query_k_list.end(), (size_t) 1000);
    } else {
        std::fill(query_k_list.begin(), query_k_list.end(), (size_t) 100);
    }
}


static void __ReadVectorQueryFromBvecs(const std::string &file_name, std::vector <VectorDataType> &query_data_list,
                                       std::vector <size_t> &query_k_list) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading vector query: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    query_data_list.clear();
    VidType vid = 0;
    while (file) {
        int32_t dim;
        file.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
        if (file.eof()) break; // End of file reached

        std::vector <uint8_t> vec(dim);
        file.read(reinterpret_cast<char *>(vec.data()), dim);
        if (!file) {
            throw std::runtime_error("Error reading file");
        }

        VectorDataType vector_data(dim, vid++);
        for (int i = 0; i < dim; ++i)
            vector_data.data[i] = vec[i];
        query_data_list.emplace_back(vector_data);
        if (vid == 100) break;
    }
    file.close();

    query_k_list.resize(query_data_list.size());
    if (file_name.find("bigann") != std::string::npos) {
        std::fill(query_k_list.begin(), query_k_list.end(), (size_t) 1000);
    } else {
        std::fill(query_k_list.begin(), query_k_list.end(), (size_t) 100);
    }
}

static void __ReadGroundTruthFromIvecs(const std::string &file_name, std::vector <std::vector<VidType>> &answer_list) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading ground truth: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    answer_list.clear();
    while (file) {
        int32_t dim;
        file.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
        if (file.eof()) break; // End of file reached

        std::vector <int32_t> vec(dim);
        file.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(int32_t));
        if (!file) {
            throw std::runtime_error("Error reading file");
        }

        std::vector <VidType> answer(dim);
        for (int i = 0; i < dim; ++i)
            answer[i] = vec[i];
        answer_list.emplace_back(answer);
    }
    file.close();
}

static void __ReadGroundTruthFromBin(const std::string &file_name, std::vector <std::vector<VidType>> &answer_list) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading ground truth: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    answer_list.clear();
    int32_t n, d;
    file.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
    file.read(reinterpret_cast<char *>(&d), sizeof(int32_t));
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::end);
    fileSize = file.tellg() - fileSize;
    assert(fileSize == n * d * (4 + 4));
    file.seekg(8, std::ios::beg);
    std::vector <int32_t> arr((n * d));
    file.read(reinterpret_cast<char *>(arr.data()), n * d * sizeof(int32_t));
    if (!file) {
        throw std::runtime_error("Error reading file");
    }
    for (int i = 0; i < n; ++i) {
        std::vector <VidType> answer(d);
        for (int j = 0; j < d; ++j) {
            answer[j] = arr[i * d + j];
        }
        answer_list.emplace_back(answer);
    }
    file.close();
}


static void __DumpVectorQueryToFvecs(const std::string &file_name, const std::vector <VectorDataType> &query_data_list,
                                     const std::vector <size_t> &query_k_list) {
    std::ofstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing vector query: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (const auto &vector_data: query_data_list) {
        int32_t dim = static_cast<int32_t>(vector_data.data.size());
        file.write(reinterpret_cast<const char *>(&dim), sizeof(int32_t));
        file.write(reinterpret_cast<const char *>(vector_data.data.data()), dim * sizeof(VectorDimensionType));
    }

    file.close();
}

static void __DumpVectorDataToBin(const std::string &file_name, const std::vector <VectorDataType> &data_list) {
    std::ofstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing vector data: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int32_t nvecs = static_cast<int32_t>(data_list.size());
    file.write(reinterpret_cast<const char *>(&nvecs), sizeof(int32_t));
    int32_t dim = static_cast<int32_t>(data_list[0].data.size());
    file.write(reinterpret_cast<const char *>(&dim), sizeof(int32_t));
    for (const auto &vector_data: data_list) {
        // file.write(reinterpret_cast<const char *>(&vector_data.vid), sizeof(VidType));
        file.write(reinterpret_cast<const char *>(vector_data.data.data()), dim * sizeof(VectorDimensionType));
    }
    std::cout << "Dumped " << data_list.size() << "embeddings into " << file_name << std::endl;
    file.close();
}

static void __DumpVectorDataToFvecs(const std::string &file_name, const std::vector <VectorDataType> &data_list) {
    std::ofstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing vector data: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }
    int i = 0;
    for (const auto &vector_data: data_list) {
        int32_t dim = static_cast<int32_t>(vector_data.data.size());
        file.write(reinterpret_cast<const char *>(&dim), sizeof(int32_t));
        file.write(reinterpret_cast<const char *>(vector_data.data.data()), dim * sizeof(VectorDimensionType));
        i++;
        // std::cout << "Writing ground truth: " << i << std::endl;
        // if (i == 100) break;
    }
    file.close();
}

static void __DumpVectorDataToFivecs(const std::string &file_name, const std::vector <VectorDataType> &data_list) {
    std::ofstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing vector data: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int32_t nvecs = static_cast<int32_t>(data_list.size());
    file.write(reinterpret_cast<const char *>(&nvecs), sizeof(int32_t));
    int32_t dim = static_cast<int32_t>(data_list[0].data.size());
    file.write(reinterpret_cast<const char *>(&dim), sizeof(int32_t));
    for (const auto &vector_data: data_list) {
        file.write(reinterpret_cast<const char *>(&vector_data.vid), sizeof(VidType));
        file.write(reinterpret_cast<const char *>(vector_data.data.data()), dim * sizeof(VectorDimensionType));
    }
    file.close();
}

static void
__DumpGroundTruthToIvecs(const std::string &file_name, const std::vector <std::vector<VidType>> &answer_list) {
    std::ofstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing ground truth: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int i = 0;
    for (const auto &answer: answer_list) {
        std::vector <int32_t> vec;
        for (auto vid: answer)
            vec.emplace_back(vid);

        int32_t k = static_cast<int32_t>(answer.size());
        file.write(reinterpret_cast<const char *>(&k), sizeof(int32_t));
        file.write(reinterpret_cast<const char *>(vec.data()), k * sizeof(int32_t));
        i++;
        std::cout << "Writing ground truth: " << i << std::endl;
        if (i == 100) break;
    }
    file.close();
}