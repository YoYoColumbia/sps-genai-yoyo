### 🔧 Model Training Instructions

This repository does not include trained model weight files (`.pth`) due to size constraints.  
To train the Energy Model and Diffusion Model on CIFAR-10:

```bash
# Activate environment and install dependencies
pip install -r requirements.txt

# Train Energy Model (EnhancedCNN)
# In usage.py, set: model_name = "Energy"
python helper_lib/usage.py

# Then train Diffusion Model
# In usage.py, set: model_name = "Diffusion"
python helper_lib/usage.py
