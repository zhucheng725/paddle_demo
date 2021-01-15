

import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer

from data_transform import TrainAugmentation
from dataloader import BasicDataLoader


from fcns import  FCN8s
from softmax_logit_loss import Basic_Loss

from utils import AverageMeter
import os
from paddle.utils.plot import Ploter


def main():
    batch_size = 4

    lr = 0.001
    num_epochs = 5
    save_epoch = 5
    image_folder = '/home/aistudio/work/dummy_data' 
    image_list_file = '/home/aistudio/work/dummy_data/list.txt'
    checkpoint_folder = './output'
    train_prompt = "train deeplab loss"

    save_flag = 1

    place = fluid.CUDAPlace(0)

    with fluid.dygraph.guard(place):

        basic_augmentation = TrainAugmentation(image_size=256)
        basic_dataloader = BasicDataLoader(image_folder=image_folder,
                                    image_list_file= image_list_file,
                                    transform=basic_augmentation,                                       
                                    shuffle=True)
        plot_cost = Ploter(train_prompt)


        model = FCN8s()


        train_dataloader = fluid.io.DataLoader.from_generator(capacity=10, use_multiprocess=True)
        train_dataloader.set_sample_generator(basic_dataloader,
                                            batch_size= batch_size,
                                            places=place)

 


        optimizer = AdamOptimizer(learning_rate=lr, parameter_list=model.parameters())
        #train_loss_meter = AverageMeter()
        step = 0 
        for epoch in range(1, num_epochs+1):

            for batch_id, data in enumerate(train_dataloader):
                image = data[0]
                label = data[1]
                
                image = fluid.layers.transpose(image, (0,3,1,2))
                pred = model(image)
                print("image.shape", image.shape, pred.shape)
                loss = Basic_Loss(pred, label)

                loss.backward()
                optimizer.minimize(loss)
                model.clear_gradients()


                loss_np = loss.numpy()[0]
                print(f"epoch:{epoch}, loss:{loss_np}")

            plot_cost.append(train_prompt, step, loss_np)
            plot_cost.plot("/home/aistudio/work/train_loss_img/train_deeplab_loss.png")

            step += 1

            if epoch %save_epoch == 0 and save_flag == 1:
                model_path = os.path.join(checkpoint_folder, f"Epoch-{epoch}-Loss-{loss_np}")

                # save model and optmizer states
                model_dict = model.state_dict()
                fluid.save_dygraph(model_dict, model_path)
                optimizer_dict = optimizer.state_dict()
                fluid.save_dygraph(optimizer_dict, model_path)
                print(f'----- Save model: {model_path}.pdparams')
                print(f'----- Save optimizer: {model_path}.pdopt')



                



if __name__ == '__main__':
    main()


