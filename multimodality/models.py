from __future__ import division
from itertools import chain
import os, sys, math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import Wav2Vec2ForCTC

from utils.parse_config import parse_model_config
from utils.utils import weights_init_normal


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    hyperparams.update({
        'batch': int(hyperparams['batch']),
        'subdivisions': int(hyperparams['subdivisions']),
        'width': int(hyperparams['width']),
        'height': int(hyperparams['height']),
        'channels': int(hyperparams['channels']),
        'optimizer': hyperparams.get('optimizer'),
        'momentum': float(hyperparams['momentum']),
        'decay': float(hyperparams['decay']),
        'learning_rate': float(hyperparams['learning_rate']),
        'burn_in': int(hyperparams['burn_in']),
        'max_batches': int(hyperparams['max_batches']),
        'policy': hyperparams['policy'],
        'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")),
                             map(float, hyperparams["scales"].split(","))))
    })
    assert hyperparams["height"] == hyperparams["width"], \
        "Height and width should be equal! Non square images are padded with zeros."
    output_filters = [hyperparams["channels"]]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}",
                                   nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            if module_def["activation"] == "mish":
                modules.add_module(f"mish_{module_i}", Mish())

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                   padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers]) // int(module_def.get("groups", 1))
            modules.add_module(f"route_{module_i}", nn.Sequential())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", nn.Sequential())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class Mish(nn.Module):
    """ The MISH activation function (https://github.com/digantamisra98/Mish) """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes):
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x, img_size):
        stride = img_size // x.size(2)
        # stride = torch.div(img_size, img_size(2), rounding_mode='trunc')
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:  # inference
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
            x[..., 4:] = x[..., 4:].sigmoid()
            x = x.view(bs, -1, self.no)

        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0]
                            for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x):
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id : group_size * (group_id + 1)] # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x = module[0](x, img_size)
                yolo_outputs.append(x)
            layer_outputs.append(x)
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)#, \
            #maps[0], maps[1], maps[2]

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


def load_weights(model_path, weights_path=None):
    """Loads the yolo model from file.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """
    print('---------------------- Loading Weights-------------------------')
    # device = torch.device("cuda" if torch.cuda.is_available()
    #                       else "cpu")  # Select device for inference
    device = 'cpu'
    model = Darknet(model_path).to(device)

    model.apply(weights_init_normal)

    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path:
        if weights_path.endswith(".pth"):
            # Load checkpoint weights
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            # Load darknet weights
            model.load_darknet_weights(weights_path)

    return model

##############################################################################################################
'''Hook module to get from the shape from (bs, 256, 80, 80) to (bs, vovab_size, 40, 40) -> (bs, 1600, vocab_size)
    This is because the Wave2vec embeddings are padded to the sequence length of 1600 across the vocab size of 33/32 for english language''''
class Hook(nn.Module):
    def __init__(self, vocab_size, encoder_layers) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.module = nn.Sequential()

        layer_defs = {'inchannels':[256,self.vocab_size],'outchannels':[self.vocab_size,self.vocab_size],
                        'kernel_size':[(3, 3),(3,3)], 'stride':[(1,1),(2,2)]}

        for idx,(i,j,k,l) in enumerate(zip(layer_defs['inchannels'],layer_defs['outchannels'],\
                                layer_defs['kernel_size'], layer_defs["stride"])):
            self.module.add_module(f"attention_conv_hook_{idx}", nn.Conv2d(i, j, kernel_size=k, stride=l, padding=1))
            self.module.add_module(f"batch_norm_attention_hook_{idx}", nn.BatchNorm2d(j, momentum=0.1, eps=1e-5))
            self.module.add_module(f"leaky_attention_hook_{idx}", nn.LeakyReLU(0.1))

        self.cross_modal = Transformer_encoder(stacked_layers=encoder_layers, embedding_size=self.vocab_size)

    def forward(self, image_embeddings, audio_embeddings):
        image_embeddings = self.module(image_embeddings).permute(0,2,3,1)
        
        ### implement new cross modal attention
        image_attention = self.cross_modal(image_embeddings.reshape([image_embeddings.shape[0], image_embeddings.shape[1]*image_embeddings.shape[2],
                                                image_embeddings.shape[3]]), audio_embeddings)
        return image_attention

'''Normalized residual block from transformer'''
class Normalized_Residual(nn.Module):
    def __init__(self, vocab_size, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.layer_norm = nn.LayerNorm(self.vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,jump,ff):
        return self.layer_norm(jump + self.dropout(ff))

'''Position-Wise-Feed-Forward Network'''
class PWFFN(nn.Module):
    def __init__(self, vocab_size, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(vocab_size, vocab_size*2)
        self.fc_2 = nn.Linear(vocab_size*2, vocab_size)      
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):       
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)       
        return x

'''Transformer style encoder to merge image and audio embeddings'''
class Transformer_encoder(nn.Module):
    def __init__(self, stacked_layers, embedding_size=32, num_heads=3, dropout=0.1):
        super().__init__()
        self.stacked_layers = stacked_layers
        self.module_dict = nn.ModuleDict({
                                    'query': nn.Linear(embedding_size, embedding_size),
                                    'key': nn.Linear(embedding_size, embedding_size),
                                    'value': nn.Linear(embedding_size, embedding_size),
                                    'MultiHeadAttention_block': nn.MultiheadAttention(embedding_size, num_heads, batch_first=True),
                                    'Add_Norm_attention': Normalized_Residual(embedding_size, dropout),
                                    'Position-Wise-Feed-Forward':PWFFN(embedding_size, dropout),
                                    'Add_Norm_FFN': Normalized_Residual(embedding_size, dropout)})

    def forward(self, image_embeddings, wave2vec_embeddings):        
#         attr_mapping = {'query':wave2vec_embeddings,'key':image_embeddings,'value':image_embeddings}
#         for i in range(self.stacked_layers):
#             qkv = []
#             for k,v in attr_mapping.items():
#                 qkv.append(self.module_dict[k](v))
#             attention_residual_norm = self.module_dict['Add_Norm_attention'](qkv[0], 
#                                                         self.module_dict['MultiHeadAttention_block'](*qkv)[0])
#             x = self.module_dict['Add_Norm_FFN'](attention_residual_norm,
#                                                 self.module_dict['Position-Wise-Feed-Forward'](attention_residual_norm))
#             for attributs in attr_mapping.keys():
#                 attr_mapping[attributs] = x
#         grid_dim = int(math.sqrt(x.shape[1]))
#         return x.reshape([x.shape[0], x.shape[2], grid_dim, grid_dim])

        attr_mapping = {'query':wave2vec_embeddings,'key':image_embeddings,'value':image_embeddings}
        for i in range(self.stacked_layers):
            qkv = []
            for k,v in attr_mapping.items():
                qkv.append(self.module_dict[k](v))
            attention_residual_norm = self.module_dict['Add_Norm_attention'](qkv[0], 
                                                        self.module_dict['MultiHeadAttention_block'](*qkv)[0])
            x = self.module_dict['Add_Norm_FFN'](attention_residual_norm,
                                                self.module_dict['Position-Wise-Feed-Forward'](attention_residual_norm))
            
            '''changed from cross attention -> self attention -> self attention -> self attention to
                            4X cross attention'''
            
            for attributs in attr_mapping.keys():
                if attributs=='query':
                  if i%2==0:
                    attr_mapping[attributs] = image_embeddings
                  else:
                    attr_mapping[attributs] = wave2vec_embeddings
                else:
                  attr_mapping[attributs] = x
        grid_dim = int(math.sqrt(x.shape[1]))
        return x.reshape([x.shape[0], x.shape[2], grid_dim, grid_dim])

'''module to get merged modalities back to the same shape expected by the 36th feature map (bs, 256, 80, 80)'''
class Attention_merge(nn.Module):
    def __init__(self, vocab_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        channel_info = [32*(2**i) for i in range(1,4)]

        self.module = nn.Sequential()
        self.module.add_module('Attention_merge_scale_1',nn.Sequential(nn.ConvTranspose2d(self.vocab_size, channel_info[0], 3, stride=2, padding=1, output_padding=1),
                                nn.BatchNorm2d(channel_info[0], momentum=0.1, eps=1e-5),
                                nn.LeakyReLU(0.1)))
        self.module.add_module('Attention_merge_scale_2',nn.Sequential(nn.Conv2d(channel_info[0], channel_info[1], 3, stride=1, padding=1),
                                nn.BatchNorm2d(channel_info[1], momentum=0.1, eps=1e-5),
                                nn.LeakyReLU(0.1)))
        self.module.add_module('Attention_merge_scale_3',nn.Sequential(nn.Conv2d(channel_info[1], channel_info[2], 3, stride=1, padding=1),
                                nn.BatchNorm2d(channel_info[2], momentum=0.1, eps=1e-5),
                                nn.LeakyReLU(0.1)))

    def forward(self,x):

        x = self.module(x)
        return x

class MODEL(nn.Module):
    def __init__(self,darknet,wave2vec,module_defs,vocab_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.seen = 0
        self.w2v2 = wave2vec
        for param in self.w2v2.parameters():
            param.requires_grad = False

        self.hook_attention = Hook(self.vocab_size,3)
        self.attention_merge = Attention_merge(self.vocab_size)

        self.layers = dict(darknet.named_modules())
        self.module_defs = module_defs[1:]
        self.module_list = nn.ModuleList(self.layers[i] for i in self.layers.keys() if len(i.split('.'))==2)
        self.yolo_layers = [layer[0]
                            for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
    def forward(self, x, audio):
        img_size = x.size(2)
        #####
        no_conv_layers = 0
                                        ######### Audio Part ##########
        ###################################################################################################
        with torch.no_grad():
            audio_features = self.w2v2(audio).logits
            padding_length = 1600 - audio_features.shape[1]
            if padding_length > 0:
                padding = (0,padding_length)
                audio_logits = F.pad(audio_features, (0, 0, padding[0], padding[1]), "constant", 0)
        ###################################################################################################

        layer_outputs, yolo_outputs, maps = [], [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)                    
            elif module_def["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id : group_size * (group_id + 1)] # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]                
                ##### used the 36th feature map to create hook for the cross attention as this is the point were the yolo heads are created
                if i==36:
                    x = self.hook_attention(x, audio_logits)
                    x = self.attention_merge(x)
            elif module_def["type"] == "yolo":
                x = module[0](x, img_size)
                yolo_outputs.append(x)
            layer_outputs.append(x)
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)

def load_model(config_path,wave2_vec_path,weights_path=None):
    model_wav2vec = Wav2Vec2ForCTC.from_pretrained(wave2_vec_path)
    model = MODEL(load_weights(config_path,weights_path),model_wav2vec,parse_model_config(config_path),33)

    return model

##############################################################################################################
