import os
import json
import shutil
import paramiko

# 读取json文件中的目标文件列表
tgt_file_list = [
    '/data1/data/train_new.json',
]
target_files = []
for tgt_file in tgt_file_list:
    with open(tgt_file, 'r') as f:
        target_files_tmp = json.load(f)
        for item in target_files_tmp:
            target_files.append(item['video']+'.mp4')

# 源服务器信息
source_server = {
    'hostname': '172.18.30.121',
    'username': 'syzhou',
    'password': "syzhou123's#1"
}

# 目标服务器信息
target_server = {
    'hostname': '172.18.30.134',
    'username': 'syzhou',
    'password': 'syzhou'
}

# 源文件夹路径
source_folder = '/data1/data/VCMR/data/activitynet/video_all_compressed'

# 目标文件夹路径
target_folder = '/raid/cfl/data/video_caption/activitycaps/video_all_compressed'

transport=paramiko.Transport(('172.18.30.134',22))
transport.connect(username='syzhou',password='syzhou')
sftp=paramiko.SFTPClient.from_transport(transport)

# 遍历源文件夹中的文件
for filename in os.listdir(source_folder):
    # 如果文件名在目标文件列表中
    if filename in target_files:
        # 构建源文件路径和目标文件路径
        source_path = os.path.join(source_folder, filename)
        print('source: ', source_path)
        target_path = os.path.join(target_folder, filename)
        print('target: ', target_path)
 
        sftp.put(source_path,target_path)

transport.close()
