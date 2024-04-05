import matplotlib.pyplot as plt
import json
import numpy as np

def compare_json_files(*file_paths):
    data = []
    labels = []

    # Đọc nội dung từ các file JSON
    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            data.append(json_data)
            labels.append(f'epoch {i+1}')

    # Tạo đồ thị
    categories = list(data[0].keys())
    num_categories = len(categories)

    # Tạo màu cho từng phần tử
    colors = plt.cm.rainbow(np.linspace(0, 1, num_categories))

    # Vẽ đồ thị
    lines = []
    for i, category in enumerate(categories):
        values = [data[j][category] for j in range(len(data))]
        line, = plt.plot(labels, values, marker='o', color=colors[i])
        lines.append(line)

    
    # Đặt tên trục và tiêu đề
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Comparison of mAP and AP for each class')
    #tạo chú thích
    short_labels = []
    for category in categories:
        if category.split('/')[-1]  == "fighter":
            short_labels.append("attacker")
        else:
            short_labels.append(category.split('/')[-1])
    #short_labels = [category.split('/')[-1] for category in categories]
    plt.legend(lines, short_labels, loc='upper left', bbox_to_anchor=(1.02, 1))

    # Hiển thị đồ thị
    plt.tight_layout()
    plt.show()


file1 = "D:/yowov2V7/YOWOv2/backup_dir/ava_v2.2/fps32_k16_bs16_large/epoch1/ava_detections.json"
file2 = "D:/yowov2V7/YOWOv2/backup_dir/ava_v2.2/fps32_k16_bs16_large/epoch2/ava_detections.json"
file3 = "D:/yowov2V7/YOWOv2/backup_dir/ava_v2.2/fps32_k16_bs16_large/epoch3/ava_detections.json"
file4 = "D:/yowov2V7/YOWOv2/backup_dir/ava_v2.2/fps32_k16_bs16_large/epoch4/ava_detections.json"
file5 = "D:/yowov2V7/YOWOv2/backup_dir/ava_v2.2/fps32_k16_bs16_large/epoch5/ava_detections.json"
compare_json_files(file1, file2, file3, file4, file5)



def compare_json(file1, file2):
    # Lấy tên các phần tử trong json1
    with open(file1, 'r') as f1:
        json1 = json.load(f1)

    # Đọc nội dung từ tệp JSON 2
    with open(file2, 'r') as f2:
        json2 = json.load(f2)
    categories = []
    for key in json1.keys():
        category = key.split('/')[-1]
        if category.split('/')[-1]  == "fighter":
            categories.append("attacker")
        else:
            categories.append(category.split('/')[-1])

    # Lấy giá trị từ json1 và json2 tương ứng với các phần tử
    values1 = [json1[key] for key in json1.keys()]
    values2 = [json2[key] for key in json2.keys()]

    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values1, label='YOWOv2')
    plt.bar(categories, values2, label='YOWO')
    #plt.xlabel('Phần tử')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Comparing YOWO and YOWOv2')

    # Thêm chú thích
    for i in range(len(categories)):
        plt.text(i, max(values1[i], values2[i]), f'{values1[i]:.3f}', ha='center', va='bottom')
        plt.text(i, min(values1[i], values2[i]), f'{values2[i]:.3f}', ha='center', va='top')

    plt.xticks(rotation=0)
    plt.show()

#file1 = "D:/yowov2V7/YOWOv2/backup_dir/ava_v2.2/epoch4/ava_detections.json"
#file1 = "D:/NO/Django_code/datasets/Tu_class_clip32s_2kf_NF_8_BS_8_10_epoch/latest_detection.json"
#compare_json(file1, file2)

def draw(file1):
    # Lấy tên các phần tử trong json1
    with open(file1, 'r') as f1:
        json1 = json.load(f1)
    categories = []
    for key in json1.keys():
        category = key.split('/')[-1]
        if category.split('/')[-1]  == "fighter":
            categories.append("attacker")
        else:
            categories.append(category.split('/')[-1])

    # Lấy giá trị từ json1 và json2 tương ứng với các phần tử
    values1 = [json1[key] for key in json1.keys()]


    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values1, label='Value AP')
    #plt.xlabel('Phần tử')
    plt.ylabel('Value')
    plt.legend()
    plt.title('mAP and AP for each class of test data')

    # Thêm chú thích
    for i in range(len(categories)):
        plt.text(i, values1[i], f'{values1[i]:.3f}', ha='center', va='bottom')

    plt.xticks(rotation=0)
    plt.show()
file_test = "D:/yowov2V7/YOWOv2/backup_dir/ava_v2.2/fps32_k16_bs16_large/test/ava_detections.json"
draw(file_test)
# image = 'C:/Users/ACER/darknet/Downloads/images.jpg'

# import cv2

# # Đọc ảnh từ file
# img = cv2.imread(image)

# # Lật ảnh theo trục tung
# flipped_img = cv2.flip(img, 1)

# # Lưu ảnh đã lật
# cv2.imwrite('flipped_image.jpg', flipped_img)

