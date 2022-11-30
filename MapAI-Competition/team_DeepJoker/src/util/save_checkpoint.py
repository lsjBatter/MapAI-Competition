import os
import yaml
import argparse
class save_checkpoint:
    def __init__(self, path, args):
        self.path = path 
        self.EPOCHS = args.EPOCHS
        self.BATCH_SIZE = args.BATCH_SIZE
        self.input = args.input
        self.output = args.output
        self.lr = args.lr
        self.crop_size = args.crop_size
        self.lidar = args.lidar
        self.backbone = args.backbone
    def save(self):
        config = {
            'EPOCH':self.EPOCHS,

            'BATCH_SIZE':self.BATCH_SIZE,

            'Backbone': self.backbone,

            'input_channles':self.input,

            'output_channles':self.output,

            'learning_rate':self.lr,

            'generate_data':{

                'crop_size':self.crop_size,
                'lidar':self.lidar

                },

        }
        # write
        with open(self.path,'w',encoding='utf-8') as f:
            yaml.dump(config,f)