import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

path = r"C:\Users\maver\PycharmProjects\pythonProject1\Sampling Data"

file_path = os.path.join(path,"data")

file_list = []
class_list = []
class_identify = []
class_name = 0

for item in os.listdir(file_path):
    file_tmp = []
    damage = os.path.join(file_path,item)

    for i in os.listdir(damage):
        file_tmp.append(os.path.join(damage,i))
    tmp = random.sample(file_tmp,2000)

    for x in tmp:
        file_list.append(x)
        class_list.append(class_name)
        class_identify.append((item))
    class_name = class_name+1


df = pd.DataFrame({"x:wave":file_list,
                   "y:label":class_list,
                   #"class":class_identify
                   })

train_dataset, test_dataset = train_test_split(df, test_size=0.2)
print(train_dataset.info())
print(test_dataset.info())

train_dataset.to_csv("train_data.csv", index=False)
test_dataset.to_csv("test_data.csv", index=False)
