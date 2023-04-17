#define MAX_TENSOR_NUM 31

typedef void* MLUaddr;

struct AddressList {
  void* addresses[MAX_TENSOR_NUM];
};

struct SizeList {
  int sizes[MAX_TENSOR_NUM];
};