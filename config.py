import os

class Config:
    def __init__(self):
        
        self.device = 'cuda:0' # ['cuda:0', 'cpu']
        self.model_type = 'transformer' # ['pose_resnet', 'resnet']

        if self.model_type == 'pose_resnet':
            self.image_resize = (256, 192)
        if self.model_type == 'resnet':
            self.image_resize = (224, 224)
        if self.model_type == 'transformer':
            self.image_resize = (224, 224)
        if self.model_type == 'vae':
            self.image_resize = (224, 224)

        self.num_epochs = 31
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.weight_decay = 1e-4
        self.num_workers = 16

        self.checkpoint_dir = 'checkpoints/experiment1'
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)