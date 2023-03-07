#include "graph/graph.h"
#include <vector>


    VertexID selectCECIStartVertex(const Graph *data_graph, const Graph *query_graph);
    void generateCECIFilterPlan(const Graph *data_graph, const Graph *query_graph, TreeNode *&tree, VertexID *&order);
    void bfsTraversal(const Graph *graph, VertexID root_vertex, TreeNode *&tree, VertexID *&bfs_order);
    void allocateBuffer(const Graph *data_graph, const Graph *query_graph, ui **&candidates, ui *&candidates_count);
    void computeCandidateWithNLF(const Graph *data_graph, const Graph *query_graph, VertexID query_vertex, ui &count, ui *buffer);
    bool CECIFilter(const Graph *data_graph, const Graph *query_graph, ui **&candidates, ui *&candidates_count,
                            ui *&order, TreeNode *&tree,  std::vector<std::unordered_map<VertexID, std::vector<VertexID >>> &TE_Candidates,
                            std::vector<std::vector<std::unordered_map<VertexID, std::vector<VertexID>>>> &NTE_Candidates);
    void generateCECIQueryPlan(const Graph *query_graph, TreeNode *tree, ui *bfs_order, ui *&order, ui *&pivot);
    void computeCandidateWithNLF(const Graph *data_graph, const Graph *query_graph, VertexID query_vertex,
                                            ui &count, ui *buffer = NULL);
    static size_t exploreCECIStyle(const Graph *data_graph, const Graph *query_graph, TreeNode *tree, ui **candidates,
                                    ui *candidates_count,
                                    std::vector<std::unordered_map<VertexID, std::vector<VertexID>>> &TE_Candidates,
                                    std::vector<std::vector<std::unordered_map<VertexID, std::vector<VertexID>>>> &NTE_Candidates,
                                    ui *order, size_t &output_limit_num, size_t &call_count);
    void compactCandidates(ui **&candidates, ui *&candidates_count, ui query_vertex_num);
    void generateValidCandidates(ui depth, ui *embedding, ui *idx_count, ui **valid_candidates, ui *order,
                                            ui *&temp_buffer, TreeNode *tree,
                                            std::vector<std::unordered_map<VertexID, std::vector<VertexID>>> &TE_Candidates,
                                            std::vector<std::vector<std::unordered_map<VertexID, std::vector<VertexID>>>> &NTE_Candidates, int max_depth);
    void sortCandidates(ui **candidates, ui *candidates_count, ui num);
