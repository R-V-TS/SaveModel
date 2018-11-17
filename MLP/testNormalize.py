from my_func.mapminmax import mapminmax
from my_func.mapminmax import back_mapminmax
from MLP.read_dataMLP import read_data


# Load Data
#metrics = ["PSNR", "PSNRHVSM", "PSNRHVS", "PSNRHMA", "PSNRHA", "FSIM", "SSIM", "MSSSIM", "GMSD", "SRSIM", "HaarPSI",
#               "VSI", "MAD_index", "DSI", "RFSIM", "GSM", "IWSSIM", "IWPSNR", "WSNR", "SFF", "IFC", "VIF", "NQM", "ADM",
#               "IGM", "PSIM", "ADDSSIM", "ADDGSIM", "DSS", "CVSSI"]
(Train_data, Train_label) = read_data(metric = 18)


#normalization

(arr, xmax, xmin) =  mapminmax(Train_data)
print(xmax, xmin)
print(arr)
arr = back_mapminmax(arr, xmax, xmin)
print(arr)

for i in range(Train_data.shape[1]):
    (Train_data[:, i], x, y) = mapminmax(Train_data[:, i])

(arr, xmax, xmin) =  mapminmax(Train_label);
print(xmax, xmin)





# Draw hist
import matplotlib.pyplot as plt


for i in range(Train_data.shape[1]):
    plt.figure(i+1)
    plt.hist(Train_data[:, i], 100)
    plt.title(("Hist for ", 1))


