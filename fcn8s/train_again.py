

import paddle.fluid as fluid
from fcns import  FCN8s
from paddle.fluid.optimizer import AdamOptimizer


place = fluid.CUDAPlace(0) 

with fluid.dygraph.guard(place):

    params_dict, opt_dict = fluid.load_dygraph("/home/aistudio/output/Epoch-5-Loss-3.9454503059387207")

    model = FCN8s()
    model.load_dict(params_dict)

    optimizer = AdamOptimizer(learning_rate=0.0001, parameter_list=model.parameters())
    optimizer.set_dict(opt_dict)
    print("load successfully")
