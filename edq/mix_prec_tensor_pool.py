import torch
import queue

class TensorPool:
    @torch.no_grad
    def __init__(self, shape, dtype) -> None:
        self.tensor_pool = []
        self.size = 0
        self.used = 0
        self.shape = shape
        self.dtype = dtype
        self.empty_queue = queue.Queue()

    def creat_tensor(self):
        # print('create_tensor')
        new_t = torch.empty(self.shape, dtype = self.dtype, device = 'cuda')
        self.tensor_pool.append(new_t)
        self.size += 1
        return new_t
    
    def get_tensor(self):
        self.used += 1
        if (self.used > self.size):
            return self.creat_tensor(), self.size-1
        cur_free = self.empty_queue.get()
        # print(f'get_tensor:{self.cur_free-1}')
        return self.tensor_pool[cur_free], cur_free

    def free_tensor(self, tidx):
        # print('free_tensor')
        self.used -= 1
        self.empty_queue.put(tidx)


class MixPrecTensorPool:
    @torch.no_grad
    def __init__(self, shape_list:list, dtype_list:list):
        self.mp_tensor_pool = {}
        for shape in shape_list:
            for dtype in dtype_list:
                self.mp_tensor_pool[(shape, dtype)] = TensorPool(shape, dtype)

    def get_tensor(self, shape, dtype):
        return self.mp_tensor_pool[(shape, dtype)].get_tensor()
    
    def free_tensor(self, shape, dtype, tidx):
        self.mp_tensor_pool[(shape, dtype)].free_tensor(tidx)

if __name__ == '__main__':
    # tensor_pool = TensorPool((512, 512), torch.int8)
    # for i in range(5):
    #     a = tensor_pool.get_tensor()
    #     print(a)
    #     exit(0)
    #     tensor_pool.get_tensor()
    #     tensor_pool.free_tensor()
    # print("-"*20)
    # for i in range(5):
    #     tensor_pool.free_tensor()
    #     tensor_pool.get_tensor()
    # print("-"*20)
    # for i in range(5):
    #     tensor_pool.get_tensor()
    shape_list = [(512,512)]
    dtype_list = [torch.int8]
    mtp = MixPrecTensorPool(shape_list, dtype_list)
    a, aid = mtp.get_tensor((512,512), torch.int8)
    print(aid)
