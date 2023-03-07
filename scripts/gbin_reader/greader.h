#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <set>
#include <vector>
#include <cstring>
#include <limits>

// #include <sys/direct.h>
// #include <sys/io.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

#define USE_DAG 1

typedef int32_t vidtype; // signed?
typedef int64_t eidtype;
typedef uint8_t vlabel_t;
typedef uint8_t elabel_t;

constexpr bool map_edges = false;
constexpr bool map_vertices = false;
constexpr bool map_vlabels = false;

class Graph
{
private:
    vlabel_t *vlabels;
    elabel_t *elabels;
    eidtype nnz; // for COO
    vidtype n_vertices;
    eidtype n_edges;
    vidtype *edges; // CSR
    eidtype *vertices; // CSR
    int vid_size;
    vidtype max_degree;
    int num_vertex_classes;

    template<typename T>
    static T* custom_alloc_global(size_t elements){
        return new T[elements];
    }
    template<typename T>
    static void read_file(std::string fname, T *& pointer, size_t elements){
        pointer = custom_alloc_global<T>(elements);
        assert(pointer);
        std::ifstream inf(fname.c_str(),std::ios::binary);
        if (!inf.good())
        {
            std::cerr << "Failed to open file: " << fname << "\n";
            exit(1);
        }
        inf.read(reinterpret_cast<char*>(pointer), sizeof(T) * elements);
        inf.close();
    }

public:
    Graph(std::string prefix, bool use_dag = false, bool has_vlabel = false);
    Graph(const Graph &)=delete;
    Graph& operator=(const Graph &)=delete;
    void orientation();
    void dump();
    ~Graph();
};


class Converter
{
private:
    vidtype *edges; // CSR
    eidtype *vertices; // CSR
    vlabel_t *vlabels;
    elabel_t *elabels;
    vidtype *odegs; // CSR
    vidtype n_vertices;
    eidtype n_edges;
    int num_vertex_classes;
    vidtype max_degree; // for meta info
    vidtype vmin;
    vidtype vmax; // not useful
    std::vector<std::pair<vidtype, vidtype>> vec_edges;
public:
    Converter(std::string input_prefix, std::string output_prefix, bool has_vlabel=false);
    vector<string> split(const string& str, const string& delim);

    ~Converter();
};

// split
vector<string> splitStr(const string& src, const string& delimiter) {
	std::vector<string> vetStr;
	
	// 入参检查
	// 1.原字符串为空或等于分隔符，返回空 vector
	if (src == "" || src == delimiter) {
		return vetStr;
	}
	// 2.分隔符为空返回单个元素为原字符串的 vector
	if (delimiter == "") {
		vetStr.push_back(src);
		return vetStr;
	}

	string::size_type startPos = 0;
	auto index = src.find(delimiter);
	while (index != string::npos) {
		auto str = src.substr(startPos, index - startPos);
		if (str != "") {
			vetStr.push_back(str);
		}
		startPos = index + delimiter.length();
		index = src.find(delimiter, startPos);
	}
	// 取最后一个子串
	auto str = src.substr(startPos);
	if (str != "") {
		vetStr.push_back(str);
	}

	return vetStr;
}


Converter::Converter(std::string input_prefix, std::string output_prefix, bool has_vlabel)
{
    std::ifstream f_stream(input_prefix.c_str(), std::ios::in);
    if(!f_stream.good()) {
        std::cerr << "Failed to open file: " << input_prefix << "\n";
        exit(1);
    };

    f_stream >> n_vertices >> n_vertices >> n_edges;

    vertices = new eidtype[n_vertices + 1];
    edges = new vidtype[n_edges];
    odegs = new vidtype[n_vertices]();

    assert(vertices);
    assert(edges);
    assert(odegs);
    char buffer[256];
    const char *delim = " ";
    char *p;
    int i = 0;
    vmin = std::numeric_limits<vidtype>::max();
    vmax = std::numeric_limits<vidtype>::min();
    while(!f_stream.eof()){
        f_stream.getline(buffer, 100);
        std::vector<string> res = splitStr(buffer, " ");
        // cout << "r" << res.size() << endl;
	    if(res.size()==2) {
            vidtype v1 = std::stoi(res[0]);
            vidtype v2 = std::stoi(res[1]);
            //odegs[v1]++; // cal degree
            vmin = std::min(vmin, std::min(v1, v2));
            vmax = std::max(vmax, std::max(v1, v2));
            vec_edges.emplace_back(v1, v2);
            ++i;
        }
    }

    // n_vertices and n_edges
    // Note: Be CAREFUL when deal with vid from origin pairs, due to start point(e.g. from 1).
    n_vertices = vmax - vmin + 1;
    n_edges = vec_edges.size();

    for (auto& item : vec_edges) {
      item.first -= vmin;
      item.second -= vmin;
    }

    // cal degree
    std::cout << "Vmin is " << vmin << "." << std::endl;
    for (int i = 0; i < vec_edges.size(); i++) {
        // cout << i << " " << vec_edges[i].first << endl;
        odegs[vec_edges[i].first]++;
        // --BEGIN odeg test--
        // for(int i = 0; i < n_vertices; ++i) {
        //     cout << odegs[i];
        // }
        // cout << endl;
        // --END odeg test--
    }
    
    // cal max degree
    max_degree = 0;
    for (int i = 0; i < n_vertices; ++i) {
        max_degree = odegs[i] > max_degree ? odegs[i] : max_degree;
    }
    std::cout << "Max degree is " << max_degree << std::endl;

    // pointer
    vertices[0] = 0;
    for (size_t i = 0; i < n_vertices; i++)
    {
        vertices[i+1] = vertices[i] + odegs[i];
        odegs[i] = 0;
    }

    // directed edges
    
    for (size_t i = 0; i < n_edges; i++){
        vidtype v0 = vec_edges[i].first;
        vidtype v1 = vec_edges[i].second;
        edges[vertices[v0] + odegs[v0]] = v1;
        odegs[v0]++;
    }

    std::cout << "CSR Transform end." << std::endl;
    /*
    //DUMP
    cout << "[DUMP] row pointer:" << endl;
    for (size_t i = 0; i < n_vertices+1; i++){
        cout << vertices[i] << " ";
    }
    cout << "\n";
    cout << "col index:" << endl;
    for (size_t i = 0; i < n_edges; i++){
        cout << edges[i] << " ";
    }
    cout << "\n";
    */

    const char *dir = output_prefix.c_str();
    if(access(dir, F_OK) == -1) {
        mkdir(dir, S_IRWXO|S_IRWXG|S_IRWXU);
        std::cout << "Directory is not exist. Creating..." << std::endl;
    } else{
        std::cout << "Founded directory." << std::endl;
    }

    std::string output_path = output_prefix + "/graph.vertex.bin";
    std::ofstream ofv(output_path, std::ios::binary);
    if (!ofv.good()) {
        std::cerr << "Failed to open file: " << output_path << "\n";
        exit(1);
    }
    ofv.write(reinterpret_cast<char*>(vertices), sizeof(eidtype) * (n_vertices+1));
    ofv.close();
    std::cout << "> Step1: vertex_bin generated.\n";
    output_path = output_prefix + "/graph.edge.bin";

    std::ofstream ofv2(output_path.c_str(), std::ios::binary);
    if (!ofv2.good()) {
        std::cerr << "Failed to open file: " << output_path << "\n";
        exit(1);
    }
    ofv2.write(reinterpret_cast<char*>(edges), sizeof(vidtype) * (n_edges));
    ofv2.close();
    std::cout << "> Step2: edge_bin generated.\n";

    output_path = output_prefix + "/graph.meta.txt";
    std::ofstream ofv3(output_path.c_str(), std::ios::out);
    if(!ofv3.good()) {
        std::cerr << "Failed to open file: " << output_path << "\n";
        exit(1);
    }
    ofv3 << n_vertices << "\n"; // ve
    ofv3 << n_edges << "\n"; // ne
    ofv3 << "4" << "\n"; // Bytes(parse)
    ofv3 << max_degree << "\n"; // max_degree
    ofv3.close();
    cout << "> Step3: graph meta info generated.\n";
    delete []vertices;
    delete []edges;
    delete []odegs;
}

Converter::~Converter()
{

}


