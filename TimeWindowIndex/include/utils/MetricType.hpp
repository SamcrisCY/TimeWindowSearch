#ifndef INDEX_METRIC_TYPE_HPP
#define INDEX_METRIC_TYPE_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>  

#include "DataType.hpp"

class MetricFunction { 
public:  
    virtual ~MetricFunction() = default; 
    virtual VectorDimensionType operator()(const VectorDataType& a, const VectorDataType& b) const = 0;  
};  
   
class EuclideanDistance : public MetricFunction {  
public:  
    VectorDimensionType operator()(const VectorDataType& a, const VectorDataType& b) const override {  
        if (a.Dimension() != b.Dimension()) {
            throw std::invalid_argument("Vector data must have the same dimension");
        }

        const size_t dim = a.Dimension();
        VectorDimensionType sum = 0.0;  
        for (size_t i = 0; i < dim; ++i) {  
            sum += (a[i] - b[i]) * (a[i] - b[i]);  
            // std::cout << i << ": " << (a[i] - b[i]) * (a[i] - b[i]) << std::endl;
        }  
        return std::sqrt(sum);  
    }  
};  

class EuclideanSquareDistance : public MetricFunction {  
public:  
    VectorDimensionType operator()(const VectorDataType& a, const VectorDataType& b) const override { 
        if (a.Dimension() != b.Dimension()) {
            throw std::invalid_argument("Vector data must have the same dimension");
        }

        const size_t dim = a.Dimension();
        VectorDimensionType sum = 0.0;  
        for (size_t i = 0; i < dim; ++i) {  
            sum += (a[i] - b[i]) * (a[i] - b[i]);  
            // std::cout << i << ": " << (a[i] - b[i]) * (a[i] - b[i]) << std::endl;
        }  
        return sum;  
    }  
};  

class InnerProductDistance : public MetricFunction {  
public:  
    VectorDimensionType operator()(const VectorDataType& a, const VectorDataType& b) const override {  
        if (a.Dimension() != b.Dimension()) {
            throw std::invalid_argument("Vector data must have the same dimension");
        }

        const size_t dim = a.Dimension();
        VectorDimensionType sum = 0.0;  
        for (size_t i = 0; i < dim; ++i) {  
            sum += a[i] * b[i];  
            // std::cout << i << ": " << a[i] * b[i] << std::endl;
        }  
        return sum;  
    }  
};  


#endif // INDEX_METRIC_TYPE_HPP