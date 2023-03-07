#include "greader.h"

int main(int argc, char *argv[]){
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <input-graph-path(MTX)> <output-dir-path>\n" << std::endl;
        std::cout << "Example: " << argv[0] << " ./datasets/5v.mtx ./5v" << std::endl;
        return 1;
    }
    std::cout << "Data reader & converter... " << std::endl;
    Converter(argv[1], argv[2], false); // input, output, MTX supported only
    std::cout << "Finished." << std::endl;
}