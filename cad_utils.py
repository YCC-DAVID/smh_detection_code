# import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import time
import code
import wandb
# import zfpy
import copy
import torch.nn as nn
import torch.nn.functional as F

subdir = f'Cmp4Train_exp/pytorch_resnet_cifar10/'
np_dir='./npdata/'
img_dir = subdir+f'visualize_info/' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weight_gradients_list = []
weight_list = [0]*16
Output_list = []
best_prec1 = 0
compress_time = [[],[]]
compress_ratio = []
commu_cost = [[],[]]
weight_dict = {}
weight_val_dict = {}

def generate_name(args):
    name = f'epo_{args.epoch}'

    if hasattr(args, 'position') and args.position:
        # name += '_fz_p'
        name += f'_fz_p_{args.position}'

    if hasattr(args, 'drop') and args.drop:
        # name += '_drp'
        name += f'_drp_{args.drop}'
        if hasattr(args, 'tolerance') and args.tolerance:
               name += f'_tol_{args.tolerance}'
        if hasattr(args, 'gamma') and args.gamma:
               name += f'_gma_{args.gamma}'
        if hasattr(args, 'metric') and args.metric:
               name += f'_m_{args.metric}'

    if hasattr(args, 'forward_hook') and args.forward_hook:
        name += '_fhook'

    if hasattr(args, 'freez_epoch') and args.freez_epoch:
        # name += '_fz_epo'
        name += f'_fz_epo_{args.freez_epoch}'

    if hasattr(args, 'compression') and args.compression:
        name += f'_cmp_{args.compression}'
        if hasattr(args, 'tolerance') and args.tolerance:
            name += f'_tol_{args.tolerance}'

    if hasattr(args, 'learning_model') and args.learning_model:
            name += f'_lm_{args.learning_model}'         
    return name


def visual_data(imgs,name):
    imgs_ =[]
    if isinstance(imgs[0],torch.Tensor):
        for i in range(len(imgs)):
            imgs_.append(imgs[i].numpy())
        imgs = imgs_
    fig,ax = plt.subplots(4,4,figsize=(12, 12))
    for i in range(4):
        for j in range(4):
            if not (isinstance(imgs[0], np.ndarray) and imgs[0].shape == (32, 32, 3)):
                ax[i,j].imshow(imgs[i*4+j].transpose(1, 2, 0))
            else:
                ax[i,j].imshow(imgs[i*4+j])
            ax[i,j].axis('off')
    plt.savefig(name)


def save_data(gradient,pos_num):
    conv = [[]for _ in range(pos_num)]
    conv_w = [[]for _ in range(pos_num)]
    # for grad in grads:
    #      gradient.extend(grad)

    for i in tqdm(range(0,len(gradient),pos_num)):
        for j in range(pos_num):
            conv[j].append(torch.mean(abs(gradient[i+j][1].clone())))
            conv_w[j].append(torch.mean(gradient[i+j][0].clone()))
        grad =[conv,conv_w]
    np.save(subdir+'gradinfo.npy',grad)


def plot_data_distribution(errdata,oridata,dcmpdata,name):
    plt.hist(oridata, bins=200, alpha=0.3, color='g',label=f'Original Data {len(oridata)}')#,density=True)
    plt.hist(dcmpdata, bins=200, alpha=0.3, color='r',label=f'Decompressed Data {len(dcmpdata)}')#,density=True)
    plt.hist(errdata, bins=200, alpha=0.8, color='b',label=f'Error Data {len(errdata)}')#,density=True)
    plt.title(name)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(name+'.png')


def save_weights_hook(grad,param,pos,wandb):
    global weight_dict,weight_val_dict
    weight_mean = torch.mean(abs(param.data.clone()))
    weight = param.data.clone()
    gradients = torch.mean(abs(grad.clone()))
    if pos not in weight_dict:
        weight_dict[pos] = weight_mean.item()
        weight_val_dict[pos] = weight
    else:
        weightdiff_ratio = (weight_mean - weight_dict[pos])/weight_dict[pos]
        weight_dis = torch.norm(weight - weight_val_dict[pos])
        weight_dict[pos]=weight_mean.item()
        # weight_val_dict[pos] = weight
        wandb.log({f'position {pos} weight difference ratio':weightdiff_ratio,f'position {pos} gradient':gradients,f'position {pos} weight distance':weight_dis})


def getparam(model,pos):
    param3 = None
    if pos == 0:
        param1 = list(model.parameters())
        param2 = None
    if pos == 1:
        param1 = list(model.layer1[0].conv1.parameters())
        param2 = list(model.layer1[0].conv2.parameters())
    if pos == 2:
        param1 = list(model.layer1[1].conv1.parameters())
        param2 = list(model.layer1[1].conv2.parameters())          
    if pos == 3:
        param1 = list(model.layer1[2].conv1.parameters())
        param2 = list(model.layer1[2].conv2.parameters())
    if pos == 4:
        param1 = list(model.layer1[3].conv1.parameters())
        param2 = list(model.layer1[3].conv2.parameters())
    if pos == 5:
        param1 = list(model.layer1[4].conv1.parameters())
        param2 = list(model.layer1[4].conv2.parameters())
    if pos == 6:
        param1 = list(model.layer2[0].conv1.parameters())
        param2 = list(model.layer2[0].conv2.parameters())
        param3 = list(model.layer2[0].downsample[0].parameters())
    if pos == 7:
        param1 = list(model.layer2[1].conv1.parameters())
        param2 = list(model.layer2[1].conv2.parameters())
    if pos == 8:
        param1 = list(model.layer2[2].conv1.parameters())
        param2 = list(model.layer2[2].conv2.parameters())
    if pos == 9:
        param1 = list(model.layer2[3].conv1.parameters())
        param2 = list(model.layer2[3].conv2.parameters())
    if pos == 10:
        param1 = list(model.layer2[4].conv1.parameters())
        param2 = list(model.layer2[4].conv2.parameters())
    if pos == 11:
        param1 = list(model.layer3[0].conv1.parameters())
        param2 = list(model.layer3[0].conv2.parameters())
        param3 = list(model.layer3[0].downsample[0].parameters())
    if pos == 12:
        param1 = list(model.layer3[1].conv1.parameters())
        param2 = list(model.layer3[1].conv2.parameters())
    if pos == 13:
        param1 = list(model.layer3[2].conv1.parameters())
        param2 = list(model.layer3[2].conv2.parameters())
    if pos == 14:
        param1 = list(model.layer3[3].conv1.parameters())
        param2 = list(model.layer3[3].conv2.parameters())
    if pos == 15:
        param1 = list(model.layer3[4].conv1.parameters())
        param2 = list(model.layer3[4].conv2.parameters())
    
    if param3 is not None:
        return param1,param2,param3
    else:
        return param1,param2,None
    

def getparam_block(model,pos):
    if pos == 0:
        param = list(model.conv1.parameters())+list(model.bn.parameters())
    if pos == 1:
        param = list(model.layer1[0].parameters())
    if pos == 2:
        param = list(model.layer1[1].parameters())
    if pos == 3:
        param = list(model.layer1[2].parameters())
    if pos == 4:
        param = list(model.layer1[3].parameters())
    if pos == 5:
        param = list(model.layer1[4].parameters())
    if pos == 6:
        param = list(model.layer2[0].parameters())
    if pos == 7:
        param = list(model.layer2[1].parameters())
    if pos == 8:
        param = list(model.layer2[2].parameters())
    if pos == 9:
        param = list(model.layer2[3].parameters())
    if pos == 10:
        param = list(model.layer2[4].parameters())
    if pos == 11:
        param = list(model.layer3[0].parameters())
    if pos == 12:
        param = list(model.layer3[1].parameters())
    if pos == 13:
        param = list(model.layer3[2].parameters())
    if pos == 14:
        param = list(model.layer3[3].parameters())
    if pos == 15:
        param = list(model.layer3[4].parameters())

    return param


def get_handle_front_res50(model,pos):
    if pos == 0:
        hook_handle = model.bn1
    elif pos == 1:
        hook_handle = model.layer1[0]
    elif pos == 2:
        hook_handle = model.layer1[1]
    elif pos == 3:
        hook_handle = model.layer1[2]
    elif pos == 4:
        hook_handle = model.layer2[0]
    elif pos == 5:
        hook_handle = model.layer2[1]
    elif pos == 6:
        hook_handle = model.layer2[2]
    elif pos == 7:
        hook_handle = model.layer2[3]
    elif pos == 8:
        hook_handle = model.layer3[0]
    elif pos == 9:
        hook_handle = model.layer3[1]
    elif pos == 10:
        hook_handle = model.layer3[2]
    elif pos == 11:
        hook_handle = model.layer3[3]
    elif pos == 12:
        hook_handle = model.layer3[4]
    elif pos == 13:
        hook_handle = model.layer3[5]
    elif pos == 14:
        hook_handle = model.layer4[0]
    elif pos == 15:
        hook_handle = model.layer4[1]
    elif pos == 16:
        hook_handle = model.layer4[2]
    return hook_handle


def getparam_block_res50(model,pos):
    if pos == 0:
        param = list(model.conv1.parameters())+list(model.bn1.parameters())
    if pos == 1:
        param = list(model.layer1.parameters())
    if pos == 2:
        param = list(model.layer2.parameters())
    if pos == 3:
        param = list(model.layer3.parameters())
    if pos == 4:
        param = list(model.layer4.parameters())

    return param


def get_handle(model,pos):
    if pos == 0:
        hook_handle = model.conv1
    elif pos == 1:
        hook_handle = model.layer1[0].conv2
    elif pos == 2:
        hook_handle = model.layer1[1].conv2
    elif pos == 3:
        hook_handle = model.layer1[2].conv2
    elif pos == 4:
        hook_handle = model.layer1[3].conv2
    elif pos == 5:
        hook_handle = model.layer1[4].conv2
    elif pos == 6:
        hook_handle = model.layer2[0].conv2
    elif pos == 7:
        hook_handle = model.layer2[1].conv2
    elif pos == 8:
        hook_handle = model.layer2[2].conv2
    elif pos == 9:
        hook_handle = model.layer2[3].conv2
    elif pos == 10:
        hook_handle = model.layer2[4].conv2
    elif pos == 11:
        hook_handle = model.layer3[0].conv2
    elif pos == 12:
        hook_handle = model.layer3[1].conv2
    elif pos == 13:
        hook_handle = model.layer3[2].conv2
    elif pos == 14:
        hook_handle = model.layer3[3].conv2
    elif pos == 15:
        hook_handle = model.layer3[4].conv2
    return hook_handle


def get_handle_front(model,pos):
    if pos == 0:
        hook_handle = model.bn
    elif pos == 1:
        hook_handle = model.layer1[0]
    elif pos == 2:
        hook_handle = model.layer1[1]
    elif pos == 3:
        hook_handle = model.layer1[2]
    elif pos == 4:
        hook_handle = model.layer1[3]
    elif pos == 5:
        hook_handle = model.layer1[4]
    elif pos == 6:
        hook_handle = model.layer2[0]
    elif pos == 7:
        hook_handle = model.layer2[1]
    elif pos == 8:
        hook_handle = model.layer2[2]
    elif pos == 9:
        hook_handle = model.layer2[3]
    elif pos == 10:
        hook_handle = model.layer2[4]
    elif pos == 11:
        hook_handle = model.layer3[0]
    elif pos == 12:
        hook_handle = model.layer3[1]
    elif pos == 13:
        hook_handle = model.layer3[2]
    elif pos == 14:
        hook_handle = model.layer3[3]
    elif pos == 15:
        hook_handle = model.layer3[4]
    return hook_handle


def check_freez_layer(model,posi):
    for pos in range(posi+1):
        if pos > 0:
            params1,params2,params3 = getparam(model,pos)
            is_frozen1 = all(not param.requires_grad for param in params1)
            is_frozen2 = all(not param.requires_grad for param in params2)
            if params3 is not None:
                is_frozen3 = all(not param.requires_grad for param in params3)
                if not is_frozen3:
                    for param3 in params3:
                        param3.requires_grad = False
                        param3.grad = None
            if not (is_frozen1 and is_frozen2):
                for param1 in params1:
                    param1.requires_grad = False
                    param1.grad = None
                for param2 in params2:
                    param2.requires_grad = False
                    param2.grad = None
            print(f'position {pos} has been freezed')    
        if pos == 0:
            params1,_,_ = getparam(model,pos)
            is_frozen1 = all(not param.requires_grad for param in params1)
            if not is_frozen1:
                for param1 in params1:
                    param1.requires_grad = False
                    param1.grad = None
            print(f'position {pos} has been freezed')
    # for pos in range(posi+1,16):
    #     if pos > 0:
    #         params1,params2,params3 = getparam(model,pos)
    #         is_active1 = all( param.requires_grad for param in params1)
    #         is_active2 = all( param.requires_grad for param in params2)
           
    #         if params3 is not None:
    #             is_active3 = all( param.requires_grad for param in params3)
    #             if not (is_active3):
    #                     for param3 in params3:
    #                         param3.requires_grad = True
    #                 # param3.grad = None
    #         if not (is_active1 and is_active2):
    #             for param1 in params1:
    #                 param1.requires_grad = True
    #                 #  param1.grad = None
    #             for param2 in params2:
    #                 param2.requires_grad = True
    #                 #  param2.grad = None
            

    return 

    # if pos < 0:
    #     return


def check_freez_block(model,posi):
    for pos in range(posi+1):
        params= getparam_block(model,pos)
        is_frozen = all(not param.requires_grad for param in params)
        if not is_frozen:
            for param in params:
                 param.requires_grad = False
                 param.grad = None
        print(f'position {pos} has been freezed') 


def check_freez_block_res50(model,posi):
    for pos in range(posi+1):
        params= getparam_block_res50(model,pos)
        is_frozen = all(not param.requires_grad for param in params)
        if not is_frozen:
            for param in params:
                 param.requires_grad = False
                 param.grad = None
        print(f'position {pos} has been freezed') 


def check_active_block(model,posi):
    for pos in range(posi,16):
        params= getparam_block(model,pos)
        is_frozen = all(param.requires_grad for param in params)
        if not is_frozen:
            for param in params:
                 param.requires_grad = True
                #  param.grad = None
            print(f'position {pos} has been unfreezed') 
    param_lin = model.linear.parameters()
    is_frozen = all(param.requires_grad for param in param_lin)
    if not is_frozen:
        for param in param_lin:
            param.requires_grad = True
        print(f'position {16} has been unfreezed')  
    return
    
  
def check_active_block_res50(model,posi):
    for pos in range(posi,5):
        params= getparam_block_res50(model,pos)
        is_frozen = all(param.requires_grad for param in params)
        if not is_frozen:
            for param in params:
                 param.requires_grad = True
                #  param.grad = None
            print(f'position {pos} has been unfreezed') 
    param_lin = model.fc.parameters()
    is_frozen = all(param.requires_grad for param in param_lin)
    if not is_frozen:
        for param in param_lin:
            param.requires_grad = True
        print(f'position {17} has been unfreezed')  
    return


def zfpy_compress_output(tol):
    def zfpy_cmp_inter(module, input, output):
        if module.training:
            trans_str= time.time()
            output = output.cpu().detach().numpy() # For training
            trans_end= time.time()
            intersize = output.nbytes
            t1 = time.time()
            compressed_data = zfpy.compress_numpy(output, tolerance=tol)
            t2 = time.time()
            intersize_cmpd = asizeof.asizeof(compressed_data)
            compress_ratio.append(intersize_cmpd/intersize)
            t3 = time.time()
            decompressed_array = zfpy.decompress_numpy(compressed_data)
            decompressed_array_cal = decompressed_array.copy()
            t4 = time.time()
            act_vle = np.mean(np.abs(output))
            noise = decompressed_array_cal - output
            # noise_mean = np.mean(np.abs(noise))
            # noise_ratio = noise_mean/act_vle
            # cos_sim = 1 - cosine(output.flatten(), decompressed_array_cal.flatten())
            wandb.log({f'cmp_ratio':intersize_cmpd/intersize,f'error ratio':noise_ratio,f'cosine similarity':cos_sim})
            # plot_data_distribution(noise.flatten(),output.flatten(),decompressed_array.flatten(),f'error_distrbutuib\Err tol {parser.parse_args().tolerance} noise ratio {noise_ratio:.2e} eucl_dis {eucl_dis:.2e} cos {cos_sim:.2e}')
            
            trans2_sta = time.time()
            output_dec = torch.from_numpy(decompressed_array).to(device)
            trans2_end = time.time()
            # code.interact(local=locals())
            compress_time[0].append(t2 - t1)
            compress_time[1].append(t4 - t3)
            commu_cost[0].append(trans_end - trans_str)
            commu_cost[1].append(trans2_end - trans2_sta)
        # print('inter data cmp and decmp has completed')
            return output_dec
    return zfpy_cmp_inter


class AvgPoolAndFlatten(nn.Module):
    def forward(self, x):
        x = F.avg_pool2d(x, x.size()[3])  # 全局池化
        x = x.view(x.size(0), -1)         # 展平
        return x
    

class Flatten(nn.Module):
    def forward(self, x):
        x = torch.flatten(x, 1)        # 展平
        return x


def seperate_model(model,pos):
    def flatten_sequential(modules):
        flattened = []
        for module in modules:
            if isinstance(module, nn.Sequential):
                flattened.extend(module.children())
            else:
                flattened.append(module)
        return flattened
    def insert_relu_after_second_layer(sequential_model):
        layers = list(sequential_model.children())  # 将原始模型的层转为列表
        if len(layers) >= 2:
            layers.insert(2, nn.ReLU())  # 在第二层之后插入 ReLU
        return nn.Sequential(*layers)
     # 初始化键名映射
    mapping = {}
    if pos == 0:
        front_layers = nn.Sequential(*copy.deepcopy(list(model.children())[:2]))
        remaining_layers = nn.Sequential(*copy.deepcopy(list(model.children())[2:]))
    elif pos == 16:
        front_layers = nn.Sequential(*copy.deepcopy(list(model.children())[:-1]))
        remaining_layers = nn.Sequential(*copy.deepcopy(list(model.children())[-1:]))
    elif pos>=1 and pos<=15:
        layer_num=(pos-1)//5
        block_num=(pos-1)%5
        if block_num == 4:
            front_layers = nn.Sequential(*copy.deepcopy(list(model.children())[:2 + layer_num+1]))
            remaining_layers = nn.Sequential(*copy.deepcopy(list(model.children())[2 + layer_num + 1:]))
        front_layers = nn.Sequential(
                *copy.deepcopy(flatten_sequential(list(model.children())[:2 + layer_num]) + 
                    flatten_sequential(list(model.children())[2 + layer_num ][:block_num + 1]))
                )
        remaining_layers = nn.Sequential(
                *copy.deepcopy(flatten_sequential(list(model.children())[2 + layer_num][block_num + 1:])
                + flatten_sequential(list(model.children())[2 + layer_num + 1:])))
        assert len(front_layers)+len(remaining_layers)==18, "seprarate model lack some part"

    front_layers = insert_relu_after_second_layer(front_layers)  # 在前半部分插入 ReLU

    # 添加键名映射
    original_state_dict = model.state_dict()
    front_state_dict = front_layers.state_dict()
    

    mapping_front = {}  # 拆分后键名 → 带前缀键名
    mapping_remain = {}  # 带前缀键名 → 原始模型键名
    mapping_final = {}  # 最终映射：拆分后键名 → 原始模型键名
    used_original_keys = set()

    # 为前半部分添加映射
    for key in front_state_dict.keys():
        prefixed_key = f"f.{key}"  # 带前缀的键名
        mapping_front[key] = prefixed_key
        for original_key in original_state_dict.keys():
            # 确保匹配 shape 且原始键未被使用
            if ( front_state_dict[key].shape == original_state_dict[original_key].shape
                and original_key not in used_original_keys
            ):
                mapping_final[prefixed_key] = original_key
                used_original_keys.add(original_key)  # 标记为已使用
                break

    

    # add the average pooling and flatten layer
    if len(remaining_layers) > 1:
        new_remaining_layers = nn.Sequential(
            *list(remaining_layers.children())[:-1],  # 剔除最后一层
            AvgPoolAndFlatten(),                      # 添加自定义模块
            list(remaining_layers.children())[-1]     # 添加最后一层
        )
    else:
        new_remaining_layers = nn.Sequential(
            AvgPoolAndFlatten(),
            *list(remaining_layers.children())
        )
        # 为后半部分添加映射

    remain_state_dict = new_remaining_layers.state_dict()
    for key in remain_state_dict.keys():
        prefixed_key = f"r.{key}"  # 带前缀的键名
        mapping_remain[key] = prefixed_key
        for original_key in original_state_dict.keys():
            if ( remain_state_dict[key].shape == original_state_dict[original_key].shape
                and original_key not in used_original_keys
            ):
                mapping_final[prefixed_key] = original_key
                used_original_keys.add(original_key)  # 标记为已使用
                break

    mapping_list=[mapping_front,mapping_remain,mapping_final]

    return front_layers,new_remaining_layers,mapping_list


def seperate_model_res50(model,pos):
    if pos == 0:
        front_layers = nn.Sequential(*copy.deepcopy(list(model.children())[:4]))
        remaining_layers = nn.Sequential(*copy.deepcopy(list(model.children())[4:]))
    elif pos >= 1 and pos <= 4:
        front_layers = nn.Sequential(*copy.deepcopy(list(model.children())[:4 + pos]))
        remaining_layers = nn.Sequential(*copy.deepcopy(list(model.children())[4 + pos:]))
    elif pos == 5:
        front_layers = nn.Sequential(*copy.deepcopy(list(model.children())[:-1]))
        remaining_layers = nn.Sequential(*copy.deepcopy(list(model.children())[-1:]))
    
    if len(remaining_layers) > 1:
        new_remaining_layers = nn.Sequential(
            *list(remaining_layers.children())[:-1],  # 剔除最后一层
            Flatten(),                      # 添加自定义模块
            list(remaining_layers.children())[-1]     # 添加最后一层
        )
    else:
        new_remaining_layers = nn.Sequential(
            Flatten(),
            *list(remaining_layers.children())
        )
        # 为后半部分添加映射
    # 添加键名映射
    original_state_dict = model.state_dict()
    front_state_dict = front_layers.state_dict()
    

    mapping_front = {}  # 拆分后键名 → 带前缀键名
    mapping_remain = {}  # 带前缀键名 → 原始模型键名
    mapping_final = {}  # 最终映射：拆分后键名 → 原始模型键名
    used_original_keys = set()

    # 为前半部分添加映射
    for key in front_state_dict.keys():
        prefixed_key = f"f.{key}"  # 带前缀的键名
        mapping_front[key] = prefixed_key
        for original_key in original_state_dict.keys():
            # 确保匹配 shape 且原始键未被使用
            if ( front_state_dict[key].shape == original_state_dict[original_key].shape
                and original_key not in used_original_keys
            ):
                mapping_final[prefixed_key] = original_key
                used_original_keys.add(original_key)  # 标记为已使用
                break


    remain_state_dict = new_remaining_layers.state_dict()
    for key in remain_state_dict.keys():
        prefixed_key = f"r.{key}"  # 带前缀的键名
        mapping_remain[key] = prefixed_key
        for original_key in original_state_dict.keys():
            if ( remain_state_dict[key].shape == original_state_dict[original_key].shape
                and original_key not in used_original_keys
            ):
                mapping_final[prefixed_key] = original_key
                used_original_keys.add(original_key)  # 标记为已使用
                break

    mapping_list=[mapping_front,mapping_remain,mapping_final]

    return front_layers,new_remaining_layers,mapping_list


def merge_models(front_model, remain_model, new_model, mappinglist):
    front_mapping, remain_mapping, final_mapping = mappinglist
    
    new_state_dict = new_model.state_dict()
    
    # 合并前半部分
    front_state_dict = front_model.state_dict()
    for split_key, prefixed_key in front_mapping.items():
        # 通过 final_mapping 找到原始模型键名
        if prefixed_key in final_mapping:
            original_key = final_mapping[prefixed_key]
            if original_key in new_state_dict:
                # 将前半部分模型的参数赋值到新模型
                new_state_dict[original_key] = front_state_dict[split_key]

    # 加载后半部分参数
    remain_state_dict = remain_model.state_dict()
    for split_key, prefixed_key in remain_mapping.items():
        # 通过 final_mapping 找到原始模型键名
        if prefixed_key in final_mapping:
            original_key = final_mapping[prefixed_key]
            if original_key in new_state_dict:
                # 将后半部分模型的参数赋值到新模型
                new_state_dict[original_key] = remain_state_dict[split_key]


    # 加载合并后的参数
    new_model.load_state_dict(new_state_dict, strict=True)
 
    return new_model


def shuffle_data(data,):
    indices = torch.randperm(data.size(0))
    data_shuffled = data[indices]
    return data_shuffled


def calculate_zfp_flops(tensor, block_size, avg_bit_length, tolerance):
    """
    计算 ZFP 压缩和解压所需的 FLOPs。

    参数：
    - tensor: torch.Tensor，输入的张量。
    - block_size: int，每个数据块的边长（如 4 表示 4x4 或 4x4x4 数据块）。
    - avg_bit_length: float，熵编码平均比特长度。
    - tolerance: float，控制量化误差。

    返回：
    - compress_flops: int，压缩所需的 FLOPs。
    - decompress_flops: int，解压所需的 FLOPs。
    """
    # 获取张量的总元素数量
    data_size = tensor.numel()

    # 数据块的大小
    block_volume = block_size ** tensor.dim()  # 适配多维张量（例如 4x4x4）

    # 数据块的数量
    num_blocks = math.ceil(data_size / block_volume)

    # FLOPs 估算
    # 离散正交变换 (DCT)：2 * N^2 * log2(N) 或更高维度扩展
    transform_flops = 2 * block_volume * math.log2(block_size)

    # 量化：N^d（块内所有元素）
    quantization_flops = block_volume

    # 熵编码：数据量大小 * 平均比特长度
    entropy_encoding_flops = block_volume * avg_bit_length

    # 压缩总 FLOPs
    compress_flops_per_block = transform_flops + quantization_flops + entropy_encoding_flops
    compress_flops = num_blocks * compress_flops_per_block

    # 解压 FLOPs
    # 逆变换与变换 FLOPs 相同；反量化 FLOPs 与量化相同
    decompress_flops_per_block = transform_flops + quantization_flops
    decompress_flops = num_blocks * decompress_flops_per_block

    return compress_flops, decompress_flops


class FrontModel(nn.Module):
    def __init__(self, patch_embed, cls_token, dist_token, pos_embed, pos_drop, blocks, split_idx):
        super(FrontModel, self).__init__()
        self.patch_embed = patch_embed
        self.cls_token = cls_token
        self.dist_token = dist_token
        self.pos_embed = pos_embed
        self.pos_drop = pos_drop
        self.blocks = blocks[:split_idx]  # 前半部分 blocks

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        return x


class RemainingModel(nn.Module):
    def __init__(self, blocks, norm, head, head_dist, split_idx):
        super(RemainingModel, self).__init__()
        self.blocks = blocks[split_idx:]
        self.norm = norm
        self.head = head
        self.head_dist = head_dist

    def forward(self, x, training=False):
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token, dist_token = x[:, 0], x[:, 1]

        x = self.head(cls_token)
        x_dist = self.head_dist(dist_token)

        if training:
            return x, x_dist
        else:
            return (x + x_dist) / 2


def seperate_model_deit(model,pos):
    # if pos == 0:
    #     front_layers = nn.Sequential(*copy.deepcopy(list(model.children())[:4]))
    #     remaining_layers = nn.Sequential(*copy.deepcopy(list(model.children())[4:]))
    # elif pos >= 1 and pos <= 11:
    #     front_layers = nn.Sequential(*(copy.deepcopy(list(model.children())[:4])+copy.deepcopy(list(model.blocks.children())[: pos-1])))
    #     remaining_layers = nn.Sequential(*(copy.deepcopy(list(model.blocks.children())[pos-1:])+copy.deepcopy(list(model.children())[5:])))
    # elif pos == 12:
    #     front_layers = nn.Sequential(*copy.deepcopy(list(model.children())[:5]))
    #     remaining_layers = nn.Sequential(*copy.deepcopy(list(model.children())[5:]))
    front_model = FrontModel(
        patch_embed=model.patch_embed,
        cls_token=model.cls_token,
        dist_token=model.dist_token,
        pos_embed=model.pos_embed,
        pos_drop=model.pos_drop,
        blocks=model.blocks,
        split_idx=pos
    )

    remaining_model = RemainingModel(
        blocks=model.blocks,
        norm=model.norm,
        head=model.head,
        head_dist=model.head_dist,
        split_idx=pos
    )
    
    original_state_dict = model.state_dict()
    front_state_dict = front_model.state_dict()
    

    mapping_front = {}  # 拆分后键名 → 带前缀键名
    mapping_remain = {}  # 带前缀键名 → 原始模型键名
    mapping_final = {}  # 最终映射：拆分后键名 → 原始模型键名
    used_original_keys = set()

    # 为前半部分添加映射
    for key in front_state_dict.keys():
        prefixed_key = f"f.{key}"  # 带前缀的键名
        mapping_front[key] = prefixed_key
        for original_key in original_state_dict.keys():
            # 确保匹配 shape 且原始键未被使用
            if ( front_state_dict[key].shape == original_state_dict[original_key].shape
                and original_key not in used_original_keys
            ):
                mapping_final[prefixed_key] = original_key
                used_original_keys.add(original_key)  # 标记为已使用
                break


    remain_state_dict = remaining_model.state_dict()
    for key in remain_state_dict.keys():
        prefixed_key = f"r.{key}"  # 带前缀的键名
        mapping_remain[key] = prefixed_key
        for original_key in original_state_dict.keys():
            if ( remain_state_dict[key].shape == original_state_dict[original_key].shape
                and original_key not in used_original_keys
            ):
                mapping_final[prefixed_key] = original_key
                used_original_keys.add(original_key)  # 标记为已使用
                break

    mapping_list=[mapping_front,mapping_remain,mapping_final]

    return front_model,remaining_model,mapping_list


# def merge_models(front_model, remain_model, new_model, mappinglist):
#     front_mapping, remain_mapping, final_mapping = mappinglist
    
#     new_state_dict = new_model.state_dict()
    
#     # 合并前半部分
#     front_state_dict = front_model.state_dict()
#     for split_key, prefixed_key in front_mapping.items():
#         # 通过 final_mapping 找到原始模型键名
#         if prefixed_key in final_mapping:
#             original_key = final_mapping[prefixed_key]
#             if original_key in new_state_dict:
#                 # 将前半部分模型的参数赋值到新模型
#                 new_state_dict[original_key] = front_state_dict[split_key]

#     # 加载后半部分参数
#     remain_state_dict = remain_model.state_dict()
#     for split_key, prefixed_key in remain_mapping.items():
#         # 通过 final_mapping 找到原始模型键名
#         if prefixed_key in final_mapping:
#             original_key = final_mapping[prefixed_key]
#             if original_key in new_state_dict:
#                 # 将后半部分模型的参数赋值到新模型
#                 new_state_dict[original_key] = remain_state_dict[split_key]


#     # 加载合并后的参数
#     new_model.load_state_dict(new_state_dict, strict=True)
 
#     return new_model


# def getparam_block_deit(model,pos):
#     if pos == 0:
#         param = [
#             model.patch_embed.parameters(),
#             [model.cls_token],  # 确保 cls_token 被包装成一个列表
#             [model.dist_token],  # 确保 dist_token 被包装成一个列表
#             [model.pos_embed],  # 确保 pos_embed 被包装成一个列表
#         ]
#         param = [p for group in param for p in group]
#     if pos == 1:
#         param = list(model.blocks[0].parameters())
#     if pos == 2:
#         param = list(model.blocks[1].parameters())
#     if pos == 3:
#         param = list(model.blocks[2].parameters())
#     if pos == 4:
#         param = list(model.blocks[3].parameters())
#     if pos == 5:
#         param = list(model.blocks[4].parameters())
#     if pos == 6:
#         param = list(model.blocks[5].parameters())
#     if pos == 7:   
#         param = list(model.blocks[6].parameters())
#     if pos == 8:
#         param = list(model.blocks[7].parameters())
#     if pos == 9:
#         param = list(model.blocks[8].parameters())
#     if pos == 10:
#         param = list(model.blocks[9].parameters())
#     if pos == 11:
#         param = list(model.blocks[10].parameters())
#     if pos == 12:
#         param = list(model.blocks[11].parameters())

#     return param

def getparam_block_deit(model, pos):
    if pos == 0:
        # 使用 model.named_parameters() 筛选对应的参数
        param = [
            model.patch_embed.parameters(),
            [model.cls_token],  # 确保 cls_token 被包装成一个列表
            [model.dist_token],  # 确保 dist_token 被包装成一个列表
            [model.pos_embed],  # 确保 pos_embed 被包装成一个列表
        ]
        param = [p for group in param for p in group]  # 展平列表
    else:
        param = list(model.blocks[pos - 1].parameters())  # 直接获取 block 的参数

    return param


def check_freez_block_deit(model,posi):

    for pos in range(posi+1):
        params= getparam_block_deit(model,pos)
        is_frozen = all(not param.requires_grad for param in params)
        if not is_frozen:
            for param in params:
                 param.requires_grad = False
                 param.grad = None
        print(f'position {pos} has been freezed') 


def choose_saved_token(activation, ratio):
    ori_len = activation.shape[0]//2
    original_token = activation[:ori_len]
    flipped_token = activation[ori_len:]
    assert original_token.shape == flipped_token.shape,"Shapes of activation and activation_flipped must match"
    activation_normalized = F.normalize(original_token, p=2, dim=2)  # 形状: (B, C, N)
    activation_flipped_normalized = F.normalize(flipped_token, p=2, dim=2)  # 形状: (B, C, N)
    similarity_matrix = torch.bmm(
        activation_normalized,  # 形状: (B, N1, C)
        activation_flipped_normalized.transpose(1, 2)  # 形状: (B, C, N2)
    )

    closest_tokens = similarity_matrix.argmax(dim=2)
    closest_similarities = similarity_matrix.gather(dim=2, index=closest_tokens.unsqueeze(-1)).squeeze(-1)
    B, N1 = closest_similarities.shape

    sorted_similarities, sorted_indices = torch.sort(closest_similarities, dim=1)
    num_to_extract = int(ratio * N1)
    least_similar_indices = sorted_indices[:, :num_to_extract]  # (B, K)
    least_similar_values = sorted_similarities[:, :num_to_extract]  # (B, K)
    

    # 获取对应的 closest_tokens 索引
    closest_tokens_for_least = closest_tokens.gather(1, least_similar_indices)  # (B, K)

    # 合并 batch 索引和 token 对索引
    batch_indices = torch.arange(B, device=activation.device).unsqueeze(-1).expand(B, num_to_extract)  # (B, K)
    least_similar_combined_indices = torch.stack(
        (batch_indices, least_similar_indices, closest_tokens_for_least), dim=-1
    )  # (B, K, 3)

    return least_similar_combined_indices, least_similar_values
    
