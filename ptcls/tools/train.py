import argparse
import os
import sys

# 添加 ptcls 到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ptcls.utils import config
from ptcls.engine.engine import Engine
from ptcls.utils import logger

def parse_args():
    parser = argparse.ArgumentParser(description='visionhub training script')
    parser.add_argument('-c', '--config', help='configuration file to use')
    parser.add_argument('-o', '--override', nargs='*', help='override configuration options')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=True)
    
    # 初始化 logger
    logger.init_logger(
        log_file=os.path.join(cfg.Global.save_inference_dir, 'train.log'),
        log_level='INFO'
    )
    logger.advertise()
    
    engine = Engine(cfg, mode="train")
    engine.train()

if __name__ == '__main__':
    main()
