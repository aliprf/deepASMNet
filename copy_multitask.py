import queue, threading, os, time
import shutil
from configuration import  IbugConf
fileQueue = queue.Queue()
destPath = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/train_before_heatmap_img'

class ThreadedCopy:
    totalFiles = 0
    copyCount = 0
    lock = threading.Lock()

    def __init__(self):
        images_dir = IbugConf.images_dir
        lbls_dir = IbugConf.lbls_dir
        fileList=[]
        for file in os.listdir(images_dir):
            if file.endswith(".jpg") or file.endswith(".png"):
                fileList.append(images_dir + '/' + file)

        if not os.path.exists(destPath):
            os.mkdir(destPath)

        self.totalFiles = len(fileList)

        print(str(self.totalFiles) + " files to copy.")
        self.threadWorkerCopy(fileList)


    def CopyWorker(self):
        while True:
            fileName = fileQueue.get()
            shutil.copy(fileName, destPath)
            fileQueue.task_done()
            with self.lock:
                self.copyCount += 1
                percent = (self.copyCount * 100) / self.totalFiles
                print(str(percent) + " percent copied.")

    def threadWorkerCopy(self, fileNameList):
        for i in range(32):
            t = threading.Thread(target=self.CopyWorker)
            t.daemon = True
            t.start()
        for fileName in fileNameList:
            fileQueue.put(fileName)
        fileQueue.join()

# ThreadedCopy()