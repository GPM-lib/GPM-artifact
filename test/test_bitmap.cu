#include <iostream>
#include <string>

#include "src/ds/bitmap.h"
#include "src/utils/launcher.h"
#include "src/utils/cuda_utils.h"

using namespace project_AntiRF;

int main(int argc, char* argv[]){
  Stream stream;
  SetDevice(0);
  Bitmap<uint8_t,3> bitmap(61);

  LOG(INFO) << "Bitmap size: " << bitmap.Size();
  LaunchKernel(stream, []__device__(dev::Bitmap<uint8_t, 3>& bitmap){
      if(TID_1D == 0){
      bitmap.query_and_mark_aotmic(3);
      bitmap.mark(9);
      }
      }, bitmap.DeviceObject());
  stream.Sync();

  bitmap.Dump(std::cout);
  LOG(INFO) << "Bitmap test done";

  return 0;
}
