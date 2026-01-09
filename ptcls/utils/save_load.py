import os
import torch
from . import logger

def _mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dygraph_pretrain(model, pretrained_path):
    if pretrained_path is None:
        return
    if not os.path.exists(pretrained_path):
        logger.error(f"Pretrained model path {pretrained_path} does not exist!")
        return
    
    state_dict = torch.load(pretrained_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    model_dict = model.state_dict()
    # 简单的权重匹配逻辑，实际移植时可能需要更复杂的映射
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                new_state_dict[k] = v
            else:
                logger.warning(f"Shape mismatch: {k}, skip loading.")
        else:
            logger.warning(f"Weight {k} not in model, skip loading.")
            
    model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"Loaded pretrained weights from {pretrained_path}")

def init_model(config, net, optimizer=None):
    checkpoints = config.get('Global', {}).get('checkpoints', None)
    if checkpoints is not None:
        load_dygraph_pretrain(net, checkpoints)
        # TODO: 加载 optimizer 状态
    
    pretrained_model = config.get('Global', {}).get('pretrained_model', None)
    if pretrained_model is not None:
        load_dygraph_pretrain(net, pretrained_model)

def save_model(net, optimizer, metric_info, model_path, model_name="", prefix='ptcls'):
    _mkdir_if_not_exist(model_path)
    save_path = os.path.join(model_path, model_name)
    state = {
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'metric_info': metric_info
    }
    torch.save(state, save_path + ".pth")
    logger.info(f"Already save model in {save_path}.pth")
