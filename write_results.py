from collections import OrderedDict

def saveImageList(images, labels, save_to_file):
     dic = OrderedDict()
     dic['image_name'] = images
     dic['category'] = np.array(labels, dtype=np.int32)
     data = pd.DataFrame(dic)
     data.to_csv(save_to_file, index=None, header=['image_name','category'])