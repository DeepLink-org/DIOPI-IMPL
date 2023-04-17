#define MAX_TENSOR_NUM 31

typedef void* MLUaddr;

struct AddressList {
    void* addresses[MAX_TENSOR_NUM];
};

struct SizeList {
    int sizes[MAX_TENSOR_NUM];
};

void bang_fused_adam_internal(AddressList grad, AddressList m, AddressList v, AddressList variable, SizeList sizes, int tensor_num, float beta1, float beta2,
                              float epsilon_correction, float learning_rate_correction, int adam_mode, float decay, float decay_correction, cnrtDim3_t k_dim,
                              cnrtFunctionType_t k_type, cnrtQueue_t queue, cnrtDataType_t cnrt_type);