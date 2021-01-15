
import paddle.fluid as fluid
import numpy as np
import time 
from paddle.fluid.dygraph import to_variable
from fcns import  FCN8s

#https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basic_concept/dygraph/DyGraph.html#id9
#原文：动态图虽然有非常多的优点，但是如果用户希望使用 C++ 部署已经训练好的模型，会存在一些不便利。
#比如，动态图中可使用 Python 原生的控制流，包含 if/else、switch、for/while，
#这些控制流需要通过一定的机制才能映射到 C++ 端，实现在 C++ 端的部署。
#使用 TracedLayer 将前向动态图模型转换为静态图模型。可以将动态图保存后做在线C++预测；
#除此以外，用户也可使用转换后的静态图模型在Python端做预测，通常比原先的动态图性能更好。

#实际测试：看不出inference model性能好，推算时间都比加载动态图要慢

'''
place = fluid.CUDAPlace(0) 
exe = fluid.Executor(place)
program, feed_vars, fetch_vars = fluid.io.load_inference_model('/home/aistudio/model', exe)
# 静态图中需要调用执行器的run方法执行计算过程

start_time = time.time()
for i in range(50):
    in_np = np.random.random([1, 3, 256, 256]).astype('float32')
    fetch, = exe.run(program, feed={feed_vars[0]: in_np}, fetch_list=fetch_vars)

end_time = time.time()
print('use time :', end_time - start_time)
print('infer: ',50/(end_time - start_time), 'hz')
#use time : 0.7896804809570312 # for basic model
#infer:  63.31674798318922 hz

#use time : 1.5242488384246826 for fcn8s model
#infer:  32.803042875647 hz
'''

place = fluid.CUDAPlace(0) 

with fluid.dygraph.guard(place):

    params_dict, opt_dict = fluid.load_dygraph("/home/aistudio/output/Epoch-5-Loss-2.9106180667877197")

    model = FCN8s()
    model.load_dict(params_dict)
    start_time = time.time()
    for i in range(50):
        in_np = np.random.random([1, 3, 256, 256]).astype('float32')
        in_var = to_variable(in_np)
        pred = model(in_var)
    end_time = time.time()
    print('use time :', end_time - start_time)
    print('infer: ',50/(end_time - start_time), 'hz')

    #use time : 0.16052722930908203 #for basic model
    #infer:  311.47363730877765 hz

    #use time : 0.7957971096038818 # for fcn8s model
    #infer:  62.83008495078367 hz







