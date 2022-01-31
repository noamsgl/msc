import scipy.io as sio

fpath = r"C:\Users\noam\Repositories\noamsgl\msc\data\seizure-detection\sample_clip.mat"

mat_contents = sio.loadmat(fpath)
print(mat_contents)
