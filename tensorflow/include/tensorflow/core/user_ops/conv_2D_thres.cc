#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

REGISTER_OP("Conv2DThres")
     .Input("input: T")
     .Input("filter: T")
     .Input("bias: T")
     .Output("output: T")
     .Attr("T: {float, double}")
     .Attr("strides: list(int)")
     .Attr("use_cudnn_on_gpu: bool = true");

using namespace tensorflow;

class Conv2DThresOp : public OpKernel {
  public:
    explicit Conv2DThresOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      // Input tensor is of the following dimensions:
      // [ batch, in_rows, in_cols, in_depth ]

      const Tensor& input = context->input(0);

      // Input filter is of the following dimensions:
      // [ filter_rows, filter_cols, in_depth, out_depth]
      const Tensor& filter = context->input(1);

      // Bias is of the following dimensions:
      // [ filter_rows, filter_cols, in_depth, out_depth]
      const Tensor& bias = context->input(2);

      // For 2D convolution, there should be 4 dimensions.
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
      OP_REQUIRES(context, filter.dims() == 4,
                  errors::InvalidArgument("filter must be 4-dimensional: ",
                                          filter.shape().DebugString()));
      
      OP_REQUIRES(context, bias.dims() == 4,
                  errors::InvalidArgument("filter must be 4-dimensional: ",
                                          filter.shape().DebugString()));

      // The last dimension for input is in_depth. It must be the same as the
      // filter's in_depth.
      const int64 in_depth = GetTensorDim(input, data_format_, 'C');
      OP_REQUIRES(
          context, in_depth == filter.dim_size(2),
          errors::InvalidArgument("input and filter must have the same depth: ",
                                  in_depth, " vs ", filter.dim_size(2)));

      // The last dimension for filter is out_depth.
      const int64 out_depth = filter.dim_size(3);

      // The second dimension for input is rows/height.
      // The first dimension for filter is rows/height.
      const int64 input_rows = GetTensorDim(input, data_format_, 'H');
      const int64 filter_rows = filter.dim_size(0);

      // The third dimension for input is columns/width.
      // The second dimension for filter is columns/width.
      const int64 input_cols = GetTensorDim(input, data_format_, 'W');
      const int64 filter_cols = filter.dim_size(1);

      // The first dimension for input is batch.
      const int64 batch = GetTensorDim(input, data_format_, 'N');

      // For now we take the stride from the second dimension only (we
      // assume row = col stride, and do not support striding on the
      // batch or depth dimension).
      const int stride = GetTensorDim(strides_, data_format_, 'H');

      int out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;

      OP_REQUIRES_OK(
          context, Get2dOutputSize(input_rows, input_cols, filter_rows,
                                   filter_cols, stride, stride, padding_,
                                   &out_rows, &out_cols, &pad_rows, &pad_cols));

      TensorShape out_shape =
        ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

      // Output tensor is of the following dimensions:
      // [ in_batch, out_rows, out_cols, out_depth ]
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    
      // Thresholding and convolution functionality
      for (int out = 0; out < out_depth; ++out) {
        int outI = 0;
        int outJ = 0;
        for (int i = 0; i < input_rows; i += stride) {
          for (int j = 0; j < input_cols; j += stride) {
            double preactivation = 0.0;
            for (int kr = 0; kr < stride; ++kr) {
              for (int kc = 0; kc < stride; ++kc) {
                for (int in = 0; in < input_depth; ++in) {
                  biased_value = input[in][i + kr][j + kc] + bias[in][kr][kc];
                  // ReLU
                  if biased_value < 0 {
                    biased_value = 0;
                  }
                  preactivation += biased_value * filter[in][kr][kc];
                }
              }
            }
            output[out][outI][outJ] = preactivation;
            ++outJ;
          }
        ++outI;
        outJ = 0;
        }
      }
    }

  private:
    std::vector<int32> strides_; 
    Padding padding_;
    TensorFormat data_format_;

    TF_DISALLOW_COPY_AND_ASSIGN(Conv2DOp);
}

REGISTER_KERNEL_BUILDER(Name("Conv2DThres").Device(DEVICE_CPU), Conv2DThresOp);
