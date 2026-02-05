#ifndef UTILS_SCALAR_TYPE_HPP
#define UTILS_SCALAR_TYPE_HPP

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
#include <map>

enum DATA_TYPE {
    STRING,
    FLOAT,
    INT
};

class attribute {
    private:
        std::string name; // attribute name
        DATA_TYPE type;

    public:
        attribute(std::string op, std::string name) {
            if (op == "string") {
                type = STRING;
            } else if (op == "float") {
                type = FLOAT;
            } else {
                type = INT;
            }
            this->name = name;
        }

        std::string getName() {
            return name;
        }

        DATA_TYPE getType() {
            return type;
        }

        virtual size_t getSize() = 0;
};

class StringType : public attribute {
    private:
        std::vector<std::string> dataSet;
        std::map<std::string, int> mp;
        std::vector<int> idSet;

    public:
        StringType(std::string name, int num) : attribute("string", name) {
            dataSet.resize(num);
            idSet.resize(num);
        } 

        void Discretization() {
            std::vector<std::string> uniqueSet(dataSet.begin(), dataSet.end());
            std::sort(uniqueSet.begin(), uniqueSet.end());
            uniqueSet.erase(std::unique(uniqueSet.begin(), uniqueSet.end()), uniqueSet.end());
            for(size_t i = 0;i < (int)uniqueSet.size();i++) {
                mp[uniqueSet[i]] = i;
            }
            for(size_t i = 0;i < dataSet.size();i++) {
                idSet[i] = mp[dataSet[i]];
            }
            return;
        }

        std::string getValue(int index) {
            return dataSet[index];
        }

        std::vector<std::string> getArray(int ptr, int len) {
            return std::vector<std::string>(dataSet.begin() + ptr, dataSet.begin() + ptr + len);
        }

        std::vector<int> getIdArray(int ptr, int len) {
            return std::vector<int>(idSet.begin() + ptr, idSet.begin() + ptr + len);
        }

        void insert(int index, std::string value) {
            dataSet[index] = value;
        }

        int getStrId(std::string str) {
            if (mp.find(str) != mp.end()) {
                return mp[str];
            } else {
                return -1; //do not have this string
            }
        }

        size_t getSize() override {
            return dataSet.size();
        }
};

class IntType : public attribute {
    private:
        std::vector<int> dataSet;

    public:
        IntType(std::string name, int num) : attribute("int", name) {
            dataSet.reserve(num);
            dataSet.resize(num);
        } 

        int getValue(int index) {
            return dataSet[index];
        }

        std::vector<int> getArray(int ptr, int len) {
            return std::vector<int>(dataSet.begin() + ptr, dataSet.begin() + ptr + len);
        }

        void insert(int index, int value) {
            dataSet[index] = value;
        } 

        size_t getSize() override {
            return dataSet.size();
        }
};

class FloatType : public attribute {
    private:
        std::vector<float> dataSet;

    public:
        FloatType(std::string name, int num) : attribute("float", name) {
            dataSet.reserve(num);
            dataSet.resize(num);
        } 

        float getValue(int index) {
            return dataSet[index];
        }

        std::vector<float> getArray(int ptr, int len) {
            return std::vector<float>(dataSet.begin() + ptr, dataSet.begin() + ptr + len);
        }

        void insert(int index, float value) {
            dataSet[index] = value;
        }

        size_t getSize() override {
            return dataSet.size();
        }
};
#endif