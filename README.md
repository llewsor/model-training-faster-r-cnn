# model-training-faster-r-cnn

python -m venv venv
venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pillow tqdm torchmetrics tensorboard pycocotools

python test_packages.py

python train.py

pip install scikit-learn matplotlib
