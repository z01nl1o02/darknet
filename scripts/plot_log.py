import os,sys
import numpy as np
from matplotlib import pyplot as plt
import argparse
import re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('log',help="log path")



class LOG:
    def __init__(self):
        pass

    def analysis(self,log_path):
        line_log = []
        re_temp_str = r'^(\d+): (\d+.\d+), (\d+.\d+) avg, (\d+.\d+) rate, '
        re_temp = re.compile(re_temp_str)
        with open(log_path,'rb') as f:
            for line in f:
                res = re_temp.findall(line)
                if len(res) == 0:
                    continue
                line = map(lambda x:float(x), res[0])
                line_log.append(line)
        self._line_log = line_log
        return
    
    def plot_one_(self,idx, desc,color):
        X = map(lambda x: int(x[0]), self._line_log)
        Y = map(lambda x: x[idx], self._line_log)
        plt.plot(X,Y,label=desc,color=color)
        return

    def plot(self):
        self.plot_one_(1,'loss','b')
        self.plot_one_(2,'loss-avg','g')
        self.plot_one_(3,'lr','r')
        plt.ylim(0,1)
        plt.legend()
        plt.show()
        return
        


if __name__=="__main__":
    args = ap.parse_args()
    log = LOG()
    log.analysis(args.log)
    log.plot()
                

