#ifndef CAFFE_DISPWRITER_LAYER_HPP_
#define CAFFE_DISPWRITER_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief DISPWriterLayer writers DISP(disparity) files
 *
 */
template<typename Dtype> 
class DISPWriterLayer : public Layer<Dtype> {
 public:
  explicit DISPWriterLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~DISPWriterLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "DISPWriter"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
 protected:
  
  void writeDispFile(string filename, const float* data, int xSize, int ySize);
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}

  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}
};

}  // namesapace caffe

#endif // CAFFE_DISP_WRITER_LAYERS_HPP_
