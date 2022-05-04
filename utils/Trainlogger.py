import os
import logging

class Logger:
    def __init__(self, args, name=__name__, filename='training_log'):
        self.args = args
        self.name = name
        self.setup(filename=filename)
    
    def setup(self, filename='training_log'):
        self.filename = filename
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')
        
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console  = logging.StreamHandler()
        console.setLevel(logging.INFO)
        
        if not os.path.exists('result/log'):
            os.makedirs('result/log')
        file_handler = logging.FileHandler('result/log/'+filename+'.log', encoding='utf8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)
        self.logger.info('PARAMETER ...')
        self.logger.info(self.args)
        # self.logger.removeHandler(console)
