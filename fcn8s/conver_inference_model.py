

from paddle.fluid.dygraph import TracedLayer
import paddle.fluid as fluid
from fcns import  FCN8s
from paddle.fluid.optimizer import AdamOptimizer
import numpy as np


place = fluid.CUDAPlace(0) 

with fluid.dygraph.guard(place):
    # 定义model的对象

    params_dict, opt_dict = fluid.load_dygraph("/home/aistudio/output/Epoch-5-Loss-2.9106180667877197")

    model = FCN8s()
    model.load_dict(params_dict)

    model.eval()

    in_np = np.random.random([1, 3, 256, 256]).astype('float32')

    input_var = fluid.dygraph.to_variable(in_np)
    pred = model(input_var)
    print('pred.shape',pred.shape)

    # 将numpy的ndarray类型的数据转换为Variable类型
    input_var = fluid.dygraph.to_variable(in_np)
    # 通过 TracerLayer.trace 接口将动态图模型转换为静态图模型
    out_dygraph, static_layer = TracedLayer.trace(model, inputs=[input_var])
    save_dirname = '/home/aistudio/model'
    # 将转换后的模型保存
    static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])
    print('save inference model successfully')




