#!/usr/bin/env python3
"""
Simple training script for H-DenseUNet
This is a simplified version of train_optimized.py for easy usage
"""

import argparse
import yaml
from train_optimized import AdvancedTrainer, load_config

def main():
    parser = argparse.ArgumentParser(description='H-DenseUNet Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override wandb setting if specified
    if args.wandb:
        config['logging']['use_wandb'] = True
    
    # Initialize and run trainer
    trainer = AdvancedTrainer(config, device=args.device)
    trainer.train()

if __name__ == '__main__':
    main()