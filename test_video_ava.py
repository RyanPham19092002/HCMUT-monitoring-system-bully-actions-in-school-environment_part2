import argparse
import cv2
import os
import time
import numpy as np
import torch
from PIL import Image
import time
from datetime import datetime
from dataset.transforms import BaseTransform
from utils.misc import load_weight
from config import build_dataset_config, build_model_config
from models import build_model



def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2')

    # basic
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--vis_thresh', default=0.1, type=float,
                        help='threshold for visualization')
    parser.add_argument('--video', default='9Y_l9NsnYE0.mp4', type=str,
                        help='AVA video name.')
    parser.add_argument('-d', '--dataset', default='ava_v2.2',
                        help='ava_v2.2')

    # model
    parser.add_argument('-v', '--version', default='yowo_v2_large', type=str,
                        help='build YOWOv2')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')
    parser.add_argument('--threshold', default=0.1, type=int,
                        help='threshold')

    return parser.parse_args()
                    

@torch.no_grad()
def run(args, d_cfg, model, device, transform, class_names):
    # path to save 

    # save_path = os.path.join(args.save_folder, 'ava_video')
    # os.makedirs(save_path, exist_ok=True)

    # # path to video
    # path_to_video = os.path.join(d_cfg['data_root'], 'videos_15min', args.video)
    path_to_video = "D:/NO/Django_code/video_test/video_13.mp4"
    
    name = path_to_video.split("/")[-1]
    # video
    video = cv2.VideoCapture(path_to_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_size = (1280, 720)
    save_name =  os.path.join("D:/YOWOv2/video_output",name)

    fps = 15.0
    out = cv2.VideoWriter(save_name, fourcc, fps, save_size)
    num_frame = 8
    start_time = time.time()
    frames=0
    count_frame = 0
    video_clip = []
    list_count_fighter = []
    alert = "Normal"
    color = (0,255,0)
    count_fight = 0
    while(True):
        fight = 0
        threat_count = 0
        ret, frame = video.read()
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S") 
        
        if ret:
            # to PIL image
            frame_pil = Image.fromarray(frame.astype(np.uint8))

            # prepare
            if len(video_clip) <= 0:
                for _ in range(num_frame):
                    video_clip.append(frame_pil)

            video_clip.append(frame_pil)
            del video_clip[0]

            # orig size
            orig_h, orig_w = frame.shape[:2]

            # transform
            x, _ = transform(video_clip)
            # List [T, 3, H, W] -> [3, T, H, W]
            x = torch.stack(x, dim=1)
            x = x.unsqueeze(0).to(device) # [B, 3, T, H, W], B=1

            t0 = time.time()
            # inference
            batch_bboxes = model(x)
            #print("inference time ", time.time() - t0, "s")

            # batch size = 1
            bboxes = batch_bboxes[0]

            # visualize detection results
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox[:4]
                det_conf = float(bbox[4])
                cls_out = [det_conf * cls_conf for cls_conf in bbox[5:]]
            
                # rescale bbox
                x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
                y1, y2 = int(y1 * orig_h), int(y2 * orig_h)

                cls_scores = np.array(cls_out)

                if max(cls_scores) < args.threshold:
                    continue

                indices = np.argmax(cls_scores)
                scores = cls_scores[indices]
                indices = [indices]
                scores = [scores]

                # indices = np.where(cls_scores > 0.0)
                # scores = cls_scores[indices]
                # indices = list(indices[0])
                # scores = list(scores)

                

                if len(scores) > 0:

                    blk   = np.zeros(frame.shape, np.uint8)
                    font  = cv2.FONT_HERSHEY_SIMPLEX
                    coord = []
                    text  = []
                    text_size = []

#-----------------------------old---------------------------------------------#
                    if indices[0]== 0:
                        fight += 1
                    # elif indices[0]==1:
                    #     threat_count+=0.5
                    else:
                        fight+=0
#-----------------------------new---------------------------------------------#
                    # if indices[0]== 0 or indices[0]==1:
                    #     fight += 1
                    # # elif indices[0]==1:
                    # #     threat_count+=0.5
                    # else:
                    #     fight+=0
                    #     # threat_count+=0

                    for _, cls_ind in enumerate(indices):
#-----------------------------old---------------------------------------------#
                        if class_names[cls_ind] == "bully":
                            # class_name = "attacker"
                            color = (0,0,255)                   
                        else:
                            class_name = class_names[cls_ind]
                            if class_name == "victim":
                                color = (255,0,0)
                            # elif class_name == "threatener":
                            #     color = (0,255,255)
                            else:
                                color = (0,255,0)
#-----------------------------new---------------------------------------------#
                        # if class_names[cls_ind] == "fighter" or class_names[cls_ind] == "threatener":
                        #     class_name = "bully"
                        #     color = (0,0,255)
                        # else:
                        #     class_name = class_names[cls_ind]
                        #     if class_name == "victim":
                        #     #     color = (255,0,0)
                        #     # elif class_name == "threatener":
                        #         color = (0,255,255)
                        #     else:
                        #         color = (0,255,0)

                        #color =  (0,255,0)
                        #print(class_names[cls_ind])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        #text.append("[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind]))
                        #text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.75, thickness=2)[0])
                        #coord.append((x1+3, y1+25+10*_))
                        #cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-20), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-8), (0, 255, 0), cv2.FILLED)
                    frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)
                    for t in range(len(text)):
                        cv2.putText(frame, text[t], coord[t], font, 0.75, (0, 0, 255), 2)
            #cv2.imshow('Model Pytorch',frame)
#-----------------------------old---------------------------------------------#
            # if fight >= 1:
            #     fight = 1
            # if threat_count >= 0.5:
            #     threat_count = 0.5
            # if (fight+threat_count) >= 1:
            #     total = 1
            # else:
            #     total = fight + threat_count
            # list_count_fighter.append(total)
#-----------------------------new---------------------------------------------#
            if fight >= 1:
                fight = 1
            # if threat_count >= 0.5:
            #     threat_count = 0.5
            # if (fight+threat_count) >= 1:
            #     total = 1
            # else:
            #     total = fight + threat_count
            list_count_fighter.append(fight)
            if len(list_count_fighter) > num_frame:
                list_count_fighter.pop(0)
            
            if len(list_count_fighter) == num_frame:
                count_fight = 0
                for i in list_count_fighter:
                    count_fight += i
#-----------------------------old---------------------------------------------#
                # if count_fight >= 4 and count_fight < 7:
                #     alert = "Threatening"
                #     color = (0,255,255)
                # elif count_fight >= 7:
                #     alert = "Bullying"
                #     color = (0,0,255)               
                # else:
                #     alert = "Normal"
                #     color = (0,255,0)
#-----------------------------new---------------------------------------------#
                # if count_fight >= 3 and count_fight < 6:
                #     alert = "Threatening"
                #     color = (0,255,255)
                if count_fight >= num_frame/2:
                    alert = "Bullying"
                    color = (0,0,255)               
                else:
                    alert = "Normal"
                    color = (0,255,0)

            #frame = cv2.resize(frame, (640, 480))  
            frames += 1
            count_frame += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            #print("elapsed_time", elapsed_time)
            if elapsed_time > 1:
                fps = frames / elapsed_time
                start_time = current_time
                frames = 0
            cv2.putText(frame, f"Time: {str(formatted_time)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) 
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Alert: {alert}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            # save
            out.write(frame)

            if args.show:
                # show
                cv2.namedWindow('key-frame detection', cv2.WINDOW_NORMAL)

                # Thay đổi kích thước cửa sổ thành (width, height)
                cv2.resizeWindow('key-frame detection', 1280, 720)

                # Hiển thị khung hình trong cửa sổ
                cv2.imshow('key-frame detection', frame)
                #cv2.imshow('key-frame detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        else:
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    class_names = d_cfg['label_map']
    num_classes = 3

    # transform
    basetransform = BaseTransform(
        img_size=d_cfg['test_size'],
        # pixel_mean=d_cfg['pixel_mean'],
        # pixel_std=d_cfg['pixel_std']
        # pixel_mean=0,
        # pixel_std=1
        )

    # build model
    model = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False
        )

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval
    model = model.to(device).eval()

    # run
    run(args=args, d_cfg=d_cfg, model=model, device=device,
        transform=basetransform, class_names=class_names)
