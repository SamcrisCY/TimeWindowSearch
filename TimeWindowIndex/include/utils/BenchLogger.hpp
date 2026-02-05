/**
	@author:	Yuxiang Zeng
	@email: 	yxzeng@buaa.edu.cn
	@date:		2024.09.28
*/
#ifndef UTILS_QUERY_LOGGER_HPP
#define UTILS_QUERY_LOGGER_HPP

#include <chrono>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

/*
Helper function: Print line number.
*/
void PrintLine(int line_number, std::ostream& os = std::cerr) {  
    os << "Line " << std::setw(3) << line_number << " --> ";  
}  

/*
Helper function: Prints a vector of floating-point values.
*/
template <typename T>
void PrintVector(const std::vector<T>& vec, std::size_t print_size = 4, int prec = 6) {
    /*
    Save the formatting information for std::cout.
    */
    std::ios old_fmt(NULL);
    old_fmt.copyfmt(std::cout);

    std::size_t slot_count = vec.size();

    std::cout << std::fixed << std::setprecision(prec);
    if (slot_count <= 2 * print_size) {
        std::cout << "(";
        for (std::size_t i = 0; i < slot_count; i++) {
            if (i > 0)
                std::cout << " ";
            std::cout << vec[i] << ((i != slot_count - 1) ? "," : ")\n");
        }
    }
    else {
        std::cout << "(";
        for (std::size_t i = 0; i < print_size; i++) {
            if (i > 0)
                std::cout << " ";
            std::cout << vec[i] << ",";
        }
        std::cout << " ...,";
        for (std::size_t i = slot_count - print_size; i < slot_count; i++) {
            std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : ")\n");
        }
    }
    std::cout << std::endl;

    /*
    Restore the old std::cout formatting.
    */
    std::cout.copyfmt(old_fmt);
}

class BenchLogger {
public:
    BenchLogger() {
        Init();
    }

    void Init() {
        queryNum = 0;
        queryTime = 0;
        queryComm = 0;    
        startTime = std::chrono::steady_clock::now();   
        endTime = startTime; 
    }

    void SetStartTimer() {
        startTime = std::chrono::steady_clock::now(); 
    }

    void SetEndTimer() {
        endTime = std::chrono::steady_clock::now(); 
    }

    void LogAddComm(double _queryComm=0.0f) {
        queryComm += _queryComm;
    }

    void LogAddTime() {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        auto _queryTime = duration.count();
        queryTime += _queryTime;
    }

    void LogOneQuery(double _queryComm=0.0f) {
        LogAddComm(_queryComm);

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        auto _queryTime = duration.count();

        queryNum += 1;
        queryTime += _queryTime;

        startTime = endTime;
    }

    std::string to_string(size_t prec=2) const {
        float AvgQueryTime = (queryNum==0) ? 0 : (queryTime/queryNum);
        float AvgQueryComm = (queryNum==0) ? 0 : (queryComm/queryNum);
        AvgQueryTime /= 1.0;
        AvgQueryComm /= 1024.0;
        std::stringstream ss;

        // ss << "-------------- Query Log --------------\n";
        ss << queryNum << " queries: runtime = " << AvgQueryTime << " [ms], communication = " << AvgQueryComm << " [KB] per query" << std::endl;

        return ss.str();
    }

    void Print() const {
        float AvgQueryTime = (queryNum==0) ? 0 : (queryTime/queryNum);
        float AvgQueryComm = (queryNum==0) ? 0 : (queryComm/queryNum);
        AvgQueryTime /= 1.0;
        AvgQueryComm /= 1024.0;
        std::cout << "-------------- Query Log --------------\n";
        std::cout << std::fixed << std::setprecision(12)
                    << queryNum << " queries: runtime = " << AvgQueryTime << " [ms], communication = " << AvgQueryComm << " [KB] per query" << std::endl;
    }

    double GetDurationTime() const {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        double queryTime = duration.count();
        return queryTime;
    }

    double GetQueryTime() const {
        return queryTime;
    }

    double GetQueryComm() const {
        return queryComm;
    }

private:
    std::chrono::steady_clock::time_point startTime, endTime;
    size_t queryNum;
    double queryTime;
    double queryComm;
};

#endif  // UTILS_QUERY_LOGGER_HPP
