#include <iostream>
#include <string>

#include "src/utils/launcher.h"
#include "src/utils/cuda_utils.h"

using namespace project_AntiRF;

int main(int argc, char* argv[]){
  Stream stream;
  SetDevice(0);
  LaunchKernel(stream, [=]__device__(){
      if(TID_1D == 0) {
      printf("Hello, GPU\n");
      }
      });
  stream.Sync();
  LOG(INFO) << "LaunchKernel test done";
  return 0;
}
