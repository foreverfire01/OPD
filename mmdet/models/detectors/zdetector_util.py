
import warnings

import torch
import mmdet

import numpy as np

from PIL import Image
import matplotlib.pylab as plt
import mmdet.models.global_variable as global_variable
from mmdet.utils import get_root_logger
import cv2
import os
import seaborn as sns

from mmdet.utils import get_root_logger


device = 'cuda:0'      # DOG


class DetectorUtil():
    def __init__(self, opd_loss):
        self.loss_opd = opd_loss
        self.logger = get_root_logger()
        super(DetectorUtil, self).__init__()


    def get_gt_center(self, img, gt_bboxes, grid_size):
        '''
            根据图像、GT框和窗口划分大小，通过中心点的方法返回特征图，每个值表明该位置目标的分类 0 无 1 有
        '''
        B, _, H, W = img.shape
        gt_list = img.new_zeros(B, H // grid_size, W // grid_size)

        for i, gt_bbox in enumerate(gt_bboxes):
            # gt_box(左上坐标，右下坐标), cx, cy 为中心点坐标
            cx = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2
            cy = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2
            gt_list[i, (cy // grid_size).long(), (cx // grid_size).long()] = 1
            
        gt_list = gt_list.view(-1)
        return gt_list

    def get_gt_square(self, img, gt_bboxes, grid_size):
        '''
            根据图像、GT框和窗口划分大小，通过gtbox覆盖区域的方法返回特征图，每个值表明该位置目标的分类 0 无 1 有
        '''
        B, _, H, W = img.shape
        gt_list = img.new_zeros(B, H // grid_size, W // grid_size)

        for i, gt_bbox in enumerate(gt_bboxes):
            for bbox in gt_bbox:
                # 获取GTbox的左上和右下坐标
                x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()

                # 将GTbox的坐标转换为对应网格的索引范围
                grid_x1 = max(0, int(x1 // grid_size)) #左上x坐标除以网格尺寸后向下取整，max0是一个防出错操作
                grid_y1 = max(0, int(y1 // grid_size))
                grid_x2 = min(W // grid_size, int((x2) // grid_size) + 1) #右下x坐标除以网格尺寸后向上取整，min(W // grid_size是一个防出错操作
                grid_y2 = min(H // grid_size, int((y2) // grid_size) + 1)

                # 在覆盖范围内的所有网格格子标记为有目标
                gt_list[i, grid_y1:grid_y2, grid_x1:grid_x2] = 1 

        gt_list = gt_list.view(-1)
        return gt_list
        

    def adjust_loss(self, predict_label, gt_label, batch_size):
        '''
            解决正负样本不均衡问题
        '''
        nonzero_index = [index for index, value in enumerate(gt_label) if value !=0]
        indextensor = torch.tensor(nonzero_index).view(-1, 1)
        extendindextensor = indextensor + torch.tensor([range(-1, 2)])        # DOG Notice!  
        extendindexlist = extendindextensor.flatten().tolist()
        unique_set = set(extendindexlist)
        unique_index = sorted(list(unique_set))
        filter_index = [x for x in unique_index if 0<= x < gt_label.shape[0]]

        temp_predict_label = predict_label[filter_index, :]
        temp_layer_gt = gt_label[filter_index]

        opd_loss = self.loss_opd(temp_predict_label, 1 - temp_layer_gt)
        return opd_loss
    

    def cal_layer_loss_statistic(self, img, gt_bboxes, opdmap, layer_index, is_calloss = False, is_adjust = False):
        '''
        '''

        img_size = img.shape[2]
        batch_size = img.shape[0]

        # 8*8  16*16
        # 4*4  8*8
        # 2*2  4*4
        # 1*1  2*2
        predict_label = opdmap.view(-1,1)

        # 28  or 48
        # 56  or 96
        # 112 or 192
        # 224 or 384
        gt_label = self.get_gt_center(img, gt_bboxes, img_size // opdmap.shape[1]).long()

        threshold = 0
        result = torch.where(opdmap[:, :, :, 0] > threshold, torch.tensor(1, device=device), torch.tensor(0, device=device))
        
        
        result = result.flatten()
        testtrue = gt_label + result
        count0 = (testtrue==0).sum().item()   # 为空对
        count2 = (testtrue==2).sum().item()   # 有船对

                                                                                                                                                                                                                                                                                                         
        global_variable.add('numshiptrue'+layer_index, count2)                    # 检测船正确的网格数 
        global_variable.add('numtrue'+layer_index, count0 + count2)               # 检测正确的网格数 
        global_variable.add('numship'+layer_index, gt_label.sum().item())         # 含船的网格数

        if (layer_index == '1'):
            global_variable.add('picsum', batch_size)  

        if (gt_label.sum().item() > 0):
            if (layer_index == '1'):
                global_variable.add('picship', 1)                                 # 含船的图片
            if (result.sum().item() > 0):
                global_variable.add('picshiptrue'+layer_index, 1)                 # 检测船正确的图片
        elif(testtrue.sum().item() == 0):
            global_variable.add('picemptytrue'+layer_index, 1)                    # 检测空正确的图片

        if (result.sum().item() > 0):
            global_variable.add('picselectasship'+layer_index, 1)                 # 检成船的图片           
    

        if is_calloss == False:
            return None

        if is_adjust == True:
            opd_loss = self.adjust_loss(predict_label, gt_label, batch_size)   #[1024, 2] [1024]
        else:
            opd_loss = self.loss_opd(predict_label, 1 - gt_label)

        return opd_loss


    def statistic_all4(self, batch_size, layer1_window_num, resetnum, is_print = False): 
        numship1 = global_variable.get_value('numship1') 
        numshiptrue1 = global_variable.get_value('numshiptrue1') 
        numtrue1 = global_variable.get_value('numtrue1') 

        numship2 = global_variable.get_value('numship2') 
        numshiptrue2 = global_variable.get_value('numshiptrue2') 
        numtrue2 = global_variable.get_value('numtrue2') 

        numship3 = global_variable.get_value('numship3') 
        numshiptrue3 = global_variable.get_value('numshiptrue3')
        numtrue3= global_variable.get_value('numtrue3')  

        numship4 = global_variable.get_value('numship4') 
        numshiptrue4 = global_variable.get_value('numshiptrue4') 
        numtrue4 = global_variable.get_value('numtrue4') 


        picsum = global_variable.get_value('picsum') 
        picship = global_variable.get_value('picship') 

        picshiptrue1 = global_variable.get_value('picshiptrue1') 
        picemptytrue1 = global_variable.get_value('picemptytrue1') 
        picselectasship1 = global_variable.get_value('picselectasship1') 

        picshiptrue2 = global_variable.get_value('picshiptrue2') 
        picemptytrue2 = global_variable.get_value('picemptytrue2') 
        picselectasship2 = global_variable.get_value('picselectasship2') 

        picshiptrue3 = global_variable.get_value('picshiptrue3') 
        picemptytrue3 = global_variable.get_value('picemptytrue3') 
        picselectasship3 = global_variable.get_value('picselectasship3') 

        picshiptrue4 = global_variable.get_value('picshiptrue4') 
        picemptytrue4 = global_variable.get_value('picemptytrue4') 
        picselectasship4 = global_variable.get_value('picselectasship4') 

        picempty = picsum - picship

        if numship1 == 0: numship1 += 0.0001
        if numship2 == 0: numship2 += 0.0001
        if numship3 == 0: numship3 += 0.0001
        if numship4 == 0: numship4 += 0.0001
        if picempty == 0: picempty += 0.0001

        if picselectasship1 == 0: picselectasship1 += 0.0001
        if picselectasship2 == 0: picselectasship2 += 0.0001
        if picselectasship3 == 0: picselectasship3 += 0.0001
        if picselectasship4 == 0: picselectasship4 += 0.0001

        if is_print:
            print( "总数:%d ---第一层:%.3f --%.3f ---第二层:%.3f --%.3f  ---第三层:%.3f --%.3f  ---第四层:%.3f --%.3f " 
                  %(picsum, 
                    numshiptrue1/numship1,
                    numtrue1/(picsum * (layer1_window_num ** 2)),
                    numshiptrue2/numship2, 
                    numtrue2/(picsum * (layer1_window_num/2) ** 2), 
                    numshiptrue3/numship3, 
                    numtrue3/(picsum * (layer1_window_num/4) ** 2), 
                    numshiptrue4/numship4, 
                    numtrue4/(picsum * (layer1_window_num/8) ** 2)
                    ))

        if picsum == resetnum:  # 4990
        # if True:  # 4990
            self.logger.info( "\n网格检测的总正确率:检测正确的网格数 / 总网格数\
                               \n网格检测船的召回率:检测船正确的网格数 / 含船的网格数\
                               \n图片检测船的正确率:检测船正确的图片 / 检测出是船的图片\
                               \n图片检测船的召回率:检测船正确的图片 / 含船的图片  \
                               \n图片检测空的召回率:检测空正确的图片 / 空的图片 \
                               \n图片检测空的召回率:检测空正确的图片 / 空的图片 \
                               \n图片总数:%d \
                               \n网格检测的总正确率  网格检测船的召回率  图片检测船的正确率 图片检测船的召回率 图片检测空的正确率  图片检测空的召回率\
                               \n第一层:%.3f        %.3f        %.3f        %.3f        %.3f        %.3f \
                               \n第二层:%.3f        %.3f        %.3f        %.3f        %.3f        %.3f \
                               \n第三层:%.3f        %.3f        %.3f        %.3f        %.3f        %.3f \
                               \n第四层:%.3f        %.3f        %.3f        %.3f        %.3f        %.3f " 
                  %(picsum, 
                    
                    numtrue1/(picsum * (layer1_window_num ** 2)),    # 网格检测总正确率：检测正确的网格数 / 总网格数
                    numshiptrue1/numship1,                           # 网格检测船的召回率：检测船正确的网格数 / 含船的网格数
                    picshiptrue1 / picselectasship1,                 # 图片检测船的正确率：检测船正确的图片 / 检测出是船的图片
                    picshiptrue1 / picship,                          # 图片检测船的召回率：检测船正确的图片 / 含船的图片
                    picemptytrue1 / (picsum -picselectasship1 + 0.0001),      # 图片检测空的正确率：检测空正确的图片 / 检测出是空的图片
                    picemptytrue1 / picempty,                        # 图片检测空的召回率：检测空正确的图片 / 空的图片
     
                    numtrue2/(picsum * (layer1_window_num/2) ** 2), 
                    numshiptrue2/numship2, 
                    picshiptrue2 / picselectasship2,                 
                    picshiptrue2 / picship,                          
                    picemptytrue2 / (picsum -picselectasship2 + 0.0001),       
                    picemptytrue2 / picempty,                        

                    numtrue3/(picsum * (layer1_window_num/4) ** 2),  
                    numshiptrue3/numship3, 
                    picshiptrue3 / picselectasship3,                 
                    picshiptrue3 / picship,                          
                    picemptytrue3 / (picsum -picselectasship3 + 0.0001),      
                    picemptytrue3 / picempty,                        

                    numtrue4/(picsum * (layer1_window_num/8) ** 2),
                    numshiptrue4/numship4, 
                    picshiptrue4 / picselectasship4,                 
                    picshiptrue4 / picship,                          
                    picemptytrue4 / (picsum -picselectasship4 + 0.0001),      
                    picemptytrue4 / picempty,                        
                    ))
            
                # 窗口数量固定
                    # %(picsum, 
                    
                    # numtrue1/(picsum * (layer1_window_num ** 2)),    # 网格检测总正确率：检测正确的网格数 / 总网格数
                    # numshiptrue1/numship1,                           # 网格检测船的召回率：检测船正确的网格数 / 含船的网格数
                    # picshiptrue1 / picselectasship1,                 # 图片检测船的正确率：检测船正确的图片 / 检测出是船的图片
                    # picshiptrue1 / picship,                          # 图片检测船的召回率：检测船正确的图片 / 含船的图片
                    # picemptytrue1 / (picsum -picselectasship1 + 0.0001),      # 图片检测空的正确率：检测空正确的图片 / 检测出是空的图片
                    # picemptytrue1 / picempty,                        # 图片检测空的召回率：检测空正确的图片 / 空的图片
     
                    # numtrue2/(picsum * (layer1_window_num) ** 2), 
                    # numshiptrue2/numship2, 
                    # picshiptrue2 / picselectasship2,                 
                    # picshiptrue2 / picship,                          
                    # picemptytrue2 / (picsum -picselectasship2 + 0.0001),       
                    # picemptytrue2 / picempty,                        

                    # numtrue3/(picsum * (layer1_window_num) ** 2),  
                    # numshiptrue3/numship3, 
                    # picshiptrue3 / picselectasship3,                 
                    # picshiptrue3 / picship,                          
                    # picemptytrue3 / (picsum -picselectasship3 + 0.0001),      
                    # picemptytrue3 / picempty,                        

                    # numtrue4/(picsum * (layer1_window_num) ** 2),
                    # numshiptrue4/numship4, 
                    # picshiptrue4 / picselectasship4,                 
                    # picshiptrue4 / picship,                          
                    # picemptytrue4 / (picsum -picselectasship4 + 0.0001),      
                    # picemptytrue4 / picempty,                        
                    # ))
            
            global_variable.reset()  



    def draw_heatmap(self, img, img_metas, featuremap, opdmap):
        img_size = img.shape[2]

        file_path = img_metas[0]['filename']
        save_dir = r'/home/cv/Data/paper/result3/' + os.path.basename(file_path)[0: -4]
        os.makedirs(save_dir, exist_ok=True)    

        plt.ion()

        # 绘制原图
        # 方法一
        # img_from_file = Image.open(img_metas[0]['filename'])
        # plt.imshow(img_from_file)

        # 方法二
        img_from_tensor = img[0].cpu().numpy().transpose(1, 2, 0)
        import mmcv
        img_restored = mmcv.imdenormalize(img_from_tensor, img_metas[0]['img_norm_cfg']['mean'], img_metas[0]['img_norm_cfg']['std'], False)
        img_restored = img_restored / (np.max(img_restored))
        plt.imshow(img_restored)
        
        alpha = 0.3 

        # 绘制one-hot热度图  
        for i in range(4):
            plt.clf()
            h = self.opdmap_2_heatmap(opdmap[i], img_size)
            h = h.cpu().numpy()
            c1 = ((1-h) * [0, 0, 0] + h * [255, 0, 0]) * alpha
            c2 = img_restored * 255 * (1 - alpha)
            superimposed_img = np.uint8(c1 + c2)
            # superimposed_img = np.uint8(c1)
            plt.imshow(superimposed_img)

            grid_num = opdmap[i].shape[1] #格子数量
            for j in range(1, grid_num):
                plt.axhline(j*img_size/grid_num)
                plt.axvline(j*img_size/grid_num)
                plt.axis('off')
                plt.show()

            plt.savefig(os.path.join(save_dir, f'binary_hot_{i}.png'), bbox_inches='tight', pad_inches=0.0, dpi= 208 )

        

        # plt.figure(figsize=(8, 8))
        # h = opdmap[2].sigmoid().view(4, 4).cpu().numpy()
        # reshapeh = h[[3, 2, 1, 0], :]
        # sns.heatmap(reshapeh, annot=True, cmap='Blues', square=True, linewidths=0.5)
        # plt.ylim(0, 4)
        # plt.show()


        # # 绘制特征图热度图
        # for i in range(len(featuremap)):
        #     plt.clf()
        #     heatmap = self.featuremap_2_heatmap(featuremap[i])
        #     heatmap = cv2.resize(heatmap, (img_size, img_size))
        #     heatmap = np.uint8(255 * heatmap)
        #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) 
        #     superimposed_img = np.uint8(img_restored * 255 * (1 - alpha) + heatmap * alpha)
        #     plt.imshow(superimposed_img)

        #     plt.savefig(os.path.join(save_dir, f'featuremap_layer{i}.png'), bbox_inches='tight', pad_inches=0.0, dpi= 208)


        # # 绘制特征图各通道热度图
        # # for c in [6,27]:#[6, 10, 22, 97, 116, 128, 27, 136]:#range(featuremap[2].shape[1]):  
        # for c in range(featuremap[3].shape[1]): 
        #     plt.clf()                          
        #     heatmap = self.featuremap_channel_2_heatmap(featuremap[3], c)  
        #     heatmap = cv2.resize(heatmap, (img_size, img_size))
        #     heatmap = np.uint8(255 * heatmap)        
        #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #     superimposed_img = np.uint8(img_restored * 255 * (1 - alpha) + heatmap * alpha)
        #     plt.imshow(superimposed_img)

        #     plt.savefig(os.path.join(save_dir, f'feature_channel{c}.png'), bbox_inches='tight', pad_inches=0.0, dpi= 208)

        plt.ioff()


    def opdmap_2_heatmap(self, opdmap, size):
        B, H, W, C = opdmap.shape

        threshold = 0
        result = torch.where(opdmap[:, :, :, 0] > threshold, torch.tensor(1, device=device), torch.tensor(0, device=device))

        x = torch.zeros(size*size, dtype=torch.uint8, device=opdmap.device)
        x = x.view(B, H, size // H, W, size // W)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        h = result.reshape(-1, H, W, 1, 1)
        x = x + h
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(size, size, 1)
        return x


    def featuremap_2_heatmap(self, feature_map):
        '''
            特征图的所有通道数值加起来，再归一化
        '''
        assert isinstance(feature_map, torch.Tensor)
        feature_map = feature_map.detach()

        heatmap = feature_map[:, 0, :] * 0
        for c in range(feature_map.shape[1]):
            heatmap += feature_map[:, c, :]
        heatmap = heatmap.cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)  # 仅降维
        
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # heatmap = sigmoid(heatmap)
        return heatmap
    

    def featuremap_channel_2_heatmap(self, feature_map, c):
        assert isinstance(feature_map, torch.Tensor)        
        heatmap = feature_map[:, c, :]
        heatmap = heatmap.cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap
