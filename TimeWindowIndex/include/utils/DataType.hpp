#ifndef UTILS_DATA_TYPE_HPP
#define UTILS_DATA_TYPE_HPP

#include <array>
#include <vector>
#include <iomanip>
#include <cstddef>
#include <stdexcept>
#include <sstream>  
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdlib>

typedef float VectorDimensionType;
typedef int VidType;

template<size_t dim>
using VectorPointType = std::array<VectorDimensionType, dim>;
   
struct VectorDataType {  
    VidType vid;  
    std::vector<VectorDimensionType> data;  

    VectorDataType(size_t dim=0, VidType _vid=0): vid(_vid) {
        data.reserve(dim);
        data.resize(dim);
        std::fill(data.begin(), data.end(), (VectorDimensionType)0);
    }

    VectorDataType(size_t dim, VidType _vid, const VectorDataType& vector_point): vid(_vid) {
        if (vector_point.Dimension() != dim) {
            throw std::invalid_argument("vector data dimension does not match");
            std::exit(EXIT_FAILURE);
        }
        data.reserve(dim);
        data.resize(dim);
        std::copy_n(vector_point.data.begin(), dim, data.begin());
    }

    VectorDataType(size_t dim, VidType _vid, const std::vector<VectorDimensionType>& arr) : vid(_vid) {
        if (arr.size() != dim) {
            throw std::invalid_argument("Data size does not match with vector data dimension");
            std::exit(EXIT_FAILURE);
        }
        data.reserve(dim);
        data.resize(dim);
        std::copy_n(arr.begin(), dim, data.begin());
    }

    void SetVid(VidType _vid) {
        vid = _vid;
    }

    void SetVectorPoint(const VectorDataType& vector_point) {
        const size_t dim = this->Dimension();
        if (vector_point.Dimension() != dim) {
            throw std::invalid_argument("vector data dimension does not match");
            std::exit(EXIT_FAILURE);
        }
        std::copy_n(vector_point.data.begin(), dim, data.begin());
    }

    void SetVectorPoint(const std::vector<VectorDimensionType>& arr) {
        const size_t dim = this->Dimension();
        if (arr.size() != dim) {
            throw std::invalid_argument("Data size does not match with vector data dimension");
            std::exit(EXIT_FAILURE);
        }
        std::copy_n(arr.begin(), dim, data.begin());
    }
  
    VectorDataType& operator=(const VectorDataType& other) {
        if (this != &other) {  
            const size_t dim = this->Dimension();
            if (other.Dimension() != dim) {
                throw std::invalid_argument("Data size does not match with vector data dimension");
                std::exit(EXIT_FAILURE);
            }
            vid = other.vid; 
            std::copy_n(other.data.begin(), dim, data.begin());
        }  
        return *this;  
    }  
  
    bool operator==(const VectorDataType& other) const {  
        if (vid != other.vid) return false;
        const size_t dim = this->Dimension();
        if (other.Dimension() != dim) return false;

        for (size_t i=0; i<dim; ++i) {
            if (other.data[i] != data[i])
                return false;
        }

        return true;  
    }  
   
    bool operator!=(const VectorDataType& other) const {  
        return !(*this == other);  
    }  
  
    VectorDimensionType& at(size_t k) {  
        if (k >= this->Dimension()) {  
            throw std::out_of_range("Index out of range");  
        }  
        return data[k];  
    }  

    const VectorDimensionType& at(size_t k) const {  
        if (k >= this->Dimension()) {  
            throw std::out_of_range("Index out of range");  
        }  
        return data[k];  
    }  
  
    VectorDimensionType& operator[](size_t k) {  
        if (k >= this->Dimension()) {  
            throw std::out_of_range("Index out of range");  
        }  
        return data[k];  
    }  

    const VectorDimensionType& operator[](size_t k) const {  
        if (k >= this->Dimension()) {  
            throw std::out_of_range("Index out of range");  
        }  
        return data[k];  
    } 

    size_t Dimension() const {
        return data.size();
    }

    std::string to_string(size_t print_size=4, size_t prec=6) const {
        const size_t dim = this->Dimension();
        std::stringstream ss;  

        ss << std::fixed << std::setprecision(prec);
        ss << "#" << vid << ": (";
        if (dim <= print_size*2) {
            for (size_t i=0; i<dim; ++i) {
                if (i > 0)
                    ss << ", " << data[i];
                else
                    ss << data[i];
            }
        } else {
            for (size_t i=0; i<print_size; ++i) {
                if (i > 0)
                    ss << ", " << data[i];
                else
                    ss << data[i];
            }
            ss << ", ...";
            for (size_t i=dim-print_size; i<dim; ++i) {
                ss << ", " << data[i];
            }
        }
        ss << ")";
        
        return ss.str();  
    } 
};

struct TimeDataType{
    VidType vid;
    double tst;
    double ten;
    VectorDataType embedding;
    TimeDataType(VidType _vid, double _tst, double _ten, VectorDataType _v) : vid(_vid), tst(_tst), ten(_ten){
        embedding.data.resize(_v.Dimension());
        embedding = _v;
        embedding.vid = _vid;
    }  
    TimeDataType(VidType _vid, double _tst) : vid(_vid), tst(_tst){
    }  
    void setEndTime(double _ten){
        ten = _ten;
    }
    TimeDataType() : vid(0), tst(0.0), ten(0.0) {}
    std::string to_string(){
        return "vid: " + std::to_string(vid) + ",  tst:" + std::to_string(tst) + ", ten:" + std::to_string(ten) + "\n embedding: " + embedding.to_string();
    }
};

#endif  // UTILS_DATA_TYPE_HPP
