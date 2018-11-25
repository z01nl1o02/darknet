import os,sys,time,argparse
from os import path as osp

sdk_root = osp.join(os.getcwd(), './')

ap = argparse.ArgumentParser()
ap.add_argument('mode',help="train/run/recall")
ap.add_argument("input_folder",help="train folder")
ap.add_argument("-weights",help="weight file",default="")





class TRAINER:
    def __init__(self):
        '''
        all file should be in input_folder
        '''
        self.class_names_file = 'classes.name'
        self.train_file = 'train.txt' 
        self.valid_file = 'valid.txt'
        self.test_file = 'test.txt'
        self.backup_folder = 'backup/'

        self.model_file = 'yolov3.cfg'


        self.weight_file = "darknet53.conv.74" #pretrained model (if not exists, training without pretrained)
        self.log_file_prefix = 'train_' #output train log file

        self.data_file = 'image.data' #this file will generated automatically
        return
    def _get_class_num(self,class_name_file):
        names = []
        with open(class_name_file, 'rb') as f:
            for line in f:
                if line == "":
                    continue
                names.append(line)
        return len(names)

    def _gen_data_file(self,input_folder):
        class_name_file = osp.join(input_folder,self.class_names_file)
        train_file = osp.join(input_folder,self.train_file)
        valid_file = osp.join(input_folder,self.valid_file)
        test_file = osp.join(input_folder,self.test_file)
        backup_folder = osp.join(input_folder,self.backup_folder)
        class_num = self._get_class_num(class_name_file)

        if not os.path.exists(backup_folder):
            os.makedirs(backup_folder)

        lines = []
        lines.append('classes = {}'.format(class_num))
        lines.append('train = {}'.format(train_file))
        lines.append('test = {}'.format(test_file))
        lines.append('valid = {}'.format(valid_file))
        lines.append('names = {}'.format(class_name_file))
        lines.append('backup = {}'.format(backup_folder))
        with open(osp.join(input_folder,self.data_file),'wb') as f:
            f.write('\n'.join(lines))
        return



    def start(self,input_folder,weight_file=""):
        self._gen_data_file(input_folder)

        log_date = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        if weight_file != "":
            if weight_file[0] != '/':
                weight_file = osp.join(input_folder,weight_file)
            if not osp.exists(weight_file):
                print("can't read {}".format(weight_file))
                return
        else:
            weight_file = self.weight_file
            weight_file = osp.join(input_folder,weight_file)
            if not osp.exists(weight_file):
                weight_file = ""

        log_file = osp.join(input_folder,"{}{}.log".format(self.log_file_prefix, log_date))
        if weight_file != "":
            cmd = '{}/darknet detector train {} {} {} 2>&1 | tee {}'.format(sdk_root,osp.join(input_folder,self.data_file),\
            osp.join(input_folder,self.model_file), osp.join(input_folder, weight_file), log_file)
        else:
            cmd = '{}/darknet detector train {} {} 2>&1 | tee {}'.format(sdk_root,osp.join(input_folder,self.data_file),\
            osp.join(input_folder,self.model_file), osp.join(input_folder,log_file))
        print(cmd)
        os.system(cmd)
        return



class TEST:
    def __init__(self):
        '''
        all file should be in input_folder
        '''
        self.class_names_file = 'classes.name'
        self.train_file = 'train.txt' 
        self.valid_file = 'valid.txt'
        self.test_file = 'test.txt'
        self.backup_folder = 'backup/'
        self.model_file = 'yolov3.cfg'
        

        

        self.data_file = 'image.data' #this file will generated automatically
        return
    def _get_class_num(self,class_name_file):
        names = []
        with open(class_name_file, 'rb') as f:
            for line in f:
                if line == "":
                    continue
                names.append(line)
        return len(names)

    def _gen_data_file(self,input_folder):
        class_name_file = osp.join(input_folder,self.class_names_file)
        train_file = osp.join(input_folder,self.train_file)
        valid_file = osp.join(input_folder,self.valid_file)
        test_file = osp.join(input_folder,self.test_file)
        backup_folder = osp.join(input_folder,self.backup_folder)
        class_num = self._get_class_num(class_name_file)

        if not os.path.exists(backup_folder):
            os.makedirs(backup_folder)

        lines = []
        lines.append('classes = {}'.format(class_num))
        lines.append('train = {}'.format(train_file))
        lines.append('test = {}'.format(test_file))
        lines.append('valid = {}'.format(valid_file))
        lines.append('names = {}'.format(class_name_file))
        lines.append('backup = {}'.format(backup_folder))
        with open(osp.join(input_folder,self.data_file),'wb') as f:
            f.write('\n'.join(lines))
        return
#./darknet detector recall2 examples/traffic/image.data examples/traffic/yolov3.cfg examples/traffic/yolov3_900.weights examples/traffic/test.txt  2>&1 | tee examples/traffic/yolov3_900.test.log


    def calc_recalling(self,input_folder,weight_file):
        self._gen_data_file(input_folder)

        if weight_file[0] != '/':
            weight_file = osp.join(input_folder,weight_file)
        if not osp.exists(weight_file):
            print("can't read {}".format(weight_file))
            return

        output_file = os.path.join(input_folder,'{}.test.log'.format(os.path.split(weight_file)[-1]))
        

        cmd = '{}/darknet detector recall2 {} {} {} {} 2>&1 | tee {}'.format(\
            sdk_root,\
            osp.join(input_folder,self.data_file),\
            osp.join(input_folder,self.model_file), \
            osp.join(input_folder, weight_file),\
            osp.join(input_folder,self.test_file),\
            output_file)
        print(cmd)
        os.system(cmd)
        return

    def run_batch_image(self,input_folder,weight_file):
        self._gen_data_file(input_folder)

        if weight_file[0] != '/':
            weight_file = osp.join(input_folder,weight_file)
        if not osp.exists(weight_file):
            print("can't read {}".format(weight_file))
            return

        output_folder = os.path.join(input_folder,'predict/')
        if not osp.exists(output_folder):
            os.makedirs(output_folder)

        cmd = '{}/darknet detector test2 {} {} {} {} -out {}'.format(\
            sdk_root,\
            osp.join(input_folder,self.data_file),\
            osp.join(input_folder,self.model_file), \
            osp.join(input_folder, weight_file),\
            osp.join(input_folder,self.test_file),\
            output_folder)
        print(cmd)
        os.system(cmd)
        return

if __name__=="__main__":
    args = ap.parse_args()
    input_folder = args.input_folder
    if input_folder[0] != '/':
        input_folder = osp.join(sdk_root,input_folder)

    if args.mode == "train":
        TRAINER().start(input_folder, args.weights)
    elif args.mode == "run":
        TEST().run_batch_image(input_folder,args.weights)
    elif args.mode == 'recall':
        TEST().calc_recalling(input_folder,args.weights)
    else:
        print 'unk mode'



