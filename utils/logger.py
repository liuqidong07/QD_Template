# -*- encoding: utf-8 -*-
'''
@File    :   logger.py
@Time    :   2022/07/06 18:02:00
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import logging
from torch.utils.tensorboard import SummaryWriter
import os
import time



class Logger(object):
    '''base logger'''

    def __init__(self, args):

        self.args = args
        self._create_logger()


    def _create_logger(self):
        '''
        Initialize the logging module. Concretely, initialize the
        tensorboard and logging
        '''
        # If demo, use default log
        if self.args.demo:
            self.args.log = False
        
        # judge whether the folder exits
        try:
            if self.args.single_test:
                main_path = r'./log/' + self.args.model_name + '/hos' + str(self.args.hos_id) + '/'
            else:
                main_path = r'./log/' + self.args.model_name + '/'
        except:
            main_path = r'./log/Pre-train/'
        if not os.path.exists(main_path):
            os.makedirs(main_path)

        # get the current time string
        now_str = time.strftime("%m%d%H%M%S", time.localtime())

        # Initialize tensorboard. Set the save folder.
        if self.args.log:
            os.makedirs(main_path + now_str + '/')
            folder_name = main_path + now_str + '/tensorboard/'
            file_path = now_str + '/bs' + str(self.args.train_batch_size) + '_therhold' + str(self.args.therhold) + '_lr' + str(self.args.learning_rate) + '.txt'
        else:
            folder_name = main_path + '/default/'
            file_path = 'default/log.txt'
        self.writer = SummaryWriter(folder_name)

        # Initialize logging. Create console and file handler
        self.logger = logging.getLogger(self.args.model_name)
        self.logger.setLevel(logging.DEBUG)  # must set
        
        # create file handler
        log_path = main_path +  file_path
        self.fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        self.fh.setLevel(logging.DEBUG)
        fm = logging.Formatter("%(asctime)s-%(message)s")
        self.fh.setFormatter(fm)
        self.logger.addHandler(self.fh)

        # record the hyper parameters in the text
        self.logger.info('The parameters are as below:')
        for kv in self.args._get_kwargs():
            self.logger.info('%s: %s' % (kv[0], str(kv[1])))
        #self.logger.info('\nStart Training:')
            
        #create console handler
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        self.logger.addHandler(self.ch)

        self.now_str = now_str

    
    def end_log(self):

        self.logger.removeHandler(self.fh)
        self.logger.removeHandler(self.ch)


    def log_metrics(self, epoch, metrics, metric_values):
        '''Write results of experiments according to your code'''
        self.logger.info('epoch: %d' % epoch)

        if self.logger:
            log_str = "Overall Results: "
            for m in metrics:
                log_str = log_str +  "\t" + m.upper() + "@" + str(self.args.topk) + ": %.4f"

            self.logger.info(log_str % tuple(metric_values))

        if self.writer:
            
            for m, mv in zip(metrics, metric_values):

                self.writer.add_scalar(m.upper()+'@'+str(self.args.topk), mv, epoch)

    
    def get_logger(self):

        try:
            return self.logger, self.writer
        except:
            raise ValueError("Please check your logger creater")

    
    def get_now_str(self):

        try:
            return self.now_str
        except:
            raise ValueError("An error occurs in logger")



class MyLogger(Logger):
    '''create your own logger'''

    def __init__(self):

        super(MyLogger, self).__init__()

    
    def _create_logger(self):
        '''You can rewrite your logger here'''

        raise NotImplementedError


    def end_log(self):
        '''Please end your logger here'''

        raise NotImplementedError
