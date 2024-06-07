from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
import datetime
import os
import random

#認証
gauth = GoogleAuth()

#ローカルWebサーバとautoを作成
#Googleドライブの認証処理
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

def exist_list(drive_folder_id):

    query = "'{}' in parents and trashed=false".format(drive_folder_id)

    exist_file = []
    file_list = drive.ListFile({'q': query }).GetList()

    for name in file_list:
        exist_file.append(name['title'])

    return exist_file

def upload (path,upload_address,item):
    f = drive.CreateFile({'title': item,
                          'parents': [{'id': upload_address}]})
    # ローカルのファイルをセットしてアップロード
    f.SetContentFile(path)
    # Googleドライブにアップロード
    f.Upload({'convert': True})

    f = None

def select(path,train_address,val_address,exist_num,exist_list,upload_num):
    i = 0
    now_upload = upload_num-exist_num

    all_list = os.listdir(path)
    now_list  = list(set(all_list)-set(exist_list))

    dataset = random.sample(now_list,now_upload)
    time = datetime.datetime.now()

    for item in dataset:

        i = i + 1
        percent = i / len(dataset) * 100
        percent = format(percent, ".2f")

        if random.random()<0.8:
            upload(os.path.join(path,item),train_address,item)
            time_2 =  datetime.datetime.now()
            a = now_upload - i
            estimate = (time_2 - time)/i
            estimate = estimate*a+datetime.datetime.now()
            print('ファイル：' + item + 'が訓練フォルダにアップロード済み。　(' + str(i) + '/' + str(len(dataset)) + ')  ' + '完成率：' + str(
                percent) + '%　現在時刻：' + time_2.strftime('%Y年%m月%d日 %H:%M:%S') + '　予測完了時刻：' + estimate.strftime('%Y年%m月%d日 %H:%M:%S'))
        else:
            upload(os.path.join(path,item),val_address,item)
            time_2 =  datetime.datetime.now()
            a = now_upload - i
            estimate = (time_2 - time)/i
            estimate = estimate*a+datetime.datetime.now()
            print('ファイル：' + item + 'が検証フォルダにアップロード済み。　(' + str(i) + '/' + str(len(dataset)) + ')  ' + '完成率：' + str(
                percent) + '%　現在時刻：' + time_2.strftime('%Y年%m月%d日 %H:%M:%S') + '　予測完了時刻：' + estimate.strftime('%Y年%m月%d日 %H:%M:%S'))


    # #ラベル名：[訓練データid,検証データid,フォルダパス]
dict = {
             "pitting" : ['1rHDEoLKIelXfAlpt5pnFKIsxB-DbI3ur','1I4Mn1UtLdxiezFOChDsNpSqefQLh8Vx0', r"C:\Users\maver\PycharmProjects\pythonProject1\data\pitting"],
             "normal"  : ['143XOHzBl1padxUMVoBhlFxHBtOQza-MB','1Jpwk-lvPV1qG7cw_1FmtrNw5XFLcVqZh',r"C:\Users\maver\PycharmProjects\pythonProject1\data\normal"]
}

upload_num = 5000

i = 0
exist_num  = []
exist = []

for a in dict:
    train_address = dict[a][0]
    val_address = dict[a][1]
    upload_path = dict[a][2]
    exist.append(exist_list(train_address),)
    exist.extend(exist_list(val_address))
    exist_num.append(len(exist_list(train_address))+len(exist_list(val_address)))
    print(a +'フォルダに' + str(len(exist_list(train_address))+len(exist_list(val_address))) + '個のファイルがいる。')

for loop in dict:
    train_address_2 = dict[loop][0]
    val_address_2 = dict[loop][1]
    upload_path_2 = dict[loop][2]
    exist_2 = exist[i]
    exist_num2 = exist_num[i]
    select(upload_path_2,train_address_2,val_address_2,exist_num2,exist_2,upload_num)
    i = i + 1
