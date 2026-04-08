import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
print(curPath, rootPath)
sys.path.append(rootPath)

from urbangpt.train.train_st_new import train

if __name__ == "__main__":
    train()
