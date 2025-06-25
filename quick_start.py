#!/usr/bin/env python3
"""
Quick Start Script for Thai Embedding Model

This script provides a simple way to get started with the Thai embedding model.
Run this to train a basic model and see it in action.
"""

import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🇹🇭 Thai Text Embedding Model - Quick Start")
    print("=" * 50)
    
    print("\n📋 Available options:")
    print("1. Train a new model")
    print("2. Evaluate existing model") 
    print("3. Run Jupyter notebook")
    print("4. Install dependencies")
    
    choice = input("\n🤔 Choose an option (1-4): ").strip()
    
    if choice == "1":
        print("\n🚀 Training a new model...")
        os.system("python scripts/train_model.py --config configs/base_config.yaml")
        
    elif choice == "2":
        model_path = input("📂 Enter model path (or press Enter for default): ").strip()
        tokenizer_path = input("📂 Enter tokenizer path (or press Enter for default): ").strip()
        
        if not model_path:
            model_path = "checkpoints/best_model.pt"
        if not tokenizer_path:
            tokenizer_path = "tokenizer/thai_tokenizer.json"
            
        print(f"\n📊 Evaluating model: {model_path}")
        os.system(f"python scripts/evaluate_model.py --model_path {model_path} --tokenizer_path {tokenizer_path}")
        
    elif choice == "3":
        print("\n📓 Opening Jupyter notebook...")
        os.system("jupyter notebook notebooks/thai_embedding_experiment.ipynb")
        
    elif choice == "4":
        print("\n📦 Installing dependencies...")
        os.system("pip install -r requirements.txt")
        
    else:
        print("❌ Invalid choice. Please choose 1-4.")
        return
    
    print("\n✅ Operation completed!")
    print("\n📚 For more details, check:")
    print("  - README.md for full documentation")
    print("  - notebooks/ for examples")
    print("  - configs/ for configuration options")

if __name__ == "__main__":
    main()
