"""
Setup Script: Initialize Knowledge Base and Train Model
Run this once to prepare the system for deployment.
"""

import os
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_knowledge_base():
    """Initialize medical knowledge base."""
    print("\n" + "="*60)
    print("STEP 1: Initializing Medical Knowledge Base")
    print("="*60 + "\n")
    
    try:
        from docs.guidelines.knowledge_base import MedicalKnowledgeBase
        
        kb = MedicalKnowledgeBase()
        kb.initialize()
        print("✓ Knowledge base initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize knowledge base: {e}")
        return False


def train_model():
    """Train the ML model."""
    print("\n" + "="*60)
    print("STEP 2: Training ML Model")
    print("="*60)
    
    try:
        from train_model import ThyroidModelTrainer
        
        trainer = ThyroidModelTrainer()
        trainer.run_full_pipeline()
        print("\n✓ Model trained and saved successfully")
        return True
    except Exception as e:
        print(f"\n❌ Failed to train model: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_output_directory():
    """Create output directory for results."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Output directory created: {output_dir}")


def main():
    """Run complete setup."""
    print("\n" + "="*80)
    print("THYROID TRIAGE AI - SYSTEM SETUP")
    print("="*80)
    
    print("\nThis script will:")
    print("  1. Initialize medical knowledge base")
    print("  2. Train ML risk prediction model")
    print("  3. Prepare system for deployment\n")
    
    # Create output directory
    create_output_directory()
    
    # Setup knowledge base
    kb_success = setup_knowledge_base()
    
    # Train model
    model_success = train_model()
    
    # Summary
    print("\n" + "="*80)
    print("SETUP SUMMARY")
    print("="*80)
    print(f"Knowledge Base: {'✓ Success' if kb_success else '❌ Failed'}")
    print(f"Model Training: {'✓ Success' if model_success else '❌ Failed'}")
    print("="*80 + "\n")
    
    if kb_success and model_success:
        print("✓ SYSTEM SETUP COMPLETE")
        print("\nNext steps:")
        print("  python src/main.py --mode demo")
        print("\nFor interactive mode:")
        print("  python src/main.py --mode interactive\n")
        return 0
    else:
        print("❌ SETUP INCOMPLETE - Some components failed")
        print("Please review errors above and try again\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
