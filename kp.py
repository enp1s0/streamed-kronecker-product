import cupy
import time

def kronecker_product_vec(A,B,X,stream=None):
    with stream:
        return cupy.reshape(cupy.matmul(cupy.matmul(A.transpose(),X),B), (-1,1))

def kronecker_product_vec_batched(lst):
    # lst = [[A,B,X,stream],...]
    res_array = []
    count = len( lst )

    for i in range(0,count):
        res_array.append(None)
        l = lst[i]
        res_array[i] = kronecker_product_vec(l[0],l[1],l[2],l[3])

    return res_array


def main():
    N = 150
    input_lst = []
    device = cupy.cuda.Device()

    for i in range(0,50):
        A = cupy.random.rand(N,N)
        B = cupy.random.rand(N,N)
        X = cupy.random.rand(N,N)
        stream = cupy.cuda.stream.Stream()

        input_lst.append([A,B,X,stream])
    # 最初
    kronecker_product_vec_batched(input_lst)
    device.synchronize()

    """
    stream = cupy.cuda.stream.Stream()
    start_time = time.time()
    for c in range(0,100):
        for l in input_lst:
            res = kronecker_product_vec(l[0],l[1],l[2],stream)
    device.synchronize()
    elapsed_time = (time.time() - start_time)/100*1000
    print("seq : elapsed time = ",elapsed_time," [ms]")
    """

    start_time = time.time()
    for c in range(0,100):
        kronecker_product_vec_batched(input_lst)
    device.synchronize()
    elapsed_time = (time.time() - start_time)/100*1000
    print("batched : elapsed time = ",elapsed_time, "[ms]")






if __name__ == "__main__":
    main()


