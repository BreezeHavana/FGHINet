import os
import pydicom
from matplotlib import pyplot
# 调用本地的 dicom file
file_num = 0
root_path = r"E:\image_aligned"
# file_name = "CAO_JIA_BIN.CT.NECK_HX_NECK_ROUTINE_C_(ADULT).0004.0001.2019.04.29.20.41.37.963400.18637985.IMA"
# file_path = os.path.join(folder_path, file_name)
# ds = pydicom.dcmread(file_path)
#
# pyplot.imshow(ds.pixel_array, cmap=pyplot.cm.bone)
# pyplot.show()
folder_names = os.listdir(root_path)
for f in folder_names:
    folder_path = os.path.join(root_path, f)
    file_names = os.listdir(folder_path)
    file_num += len(file_names)
print(file_num)