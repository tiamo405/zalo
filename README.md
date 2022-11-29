👋 Hi, I’m @tiamo405
# Challenge.zalo.ai
# Nautilus 
## Tham khảo code trên colab hoặc clone và làm theo hướng dẫn ở dưới
## [LINK COLAB](https://colab.research.google.com/drive/1mpuLuz49P_H5J0-VdcraaLVmRg7GlFx-?usp=sharing)
- B1: git clone  https://github.com/tiamo405/zalo.git
- B2. terminal
    ```sh
    pip install -r requirements.txt
    mkdir zalo/dataset
    mkdir zalo/dataset/train
    mkdir zalo/dataset/test
    mkdir zalo/saved_models
    ```
- B3. Download data train, test

    - [data train](https://drive.google.com/file/d/1tjMZkc7YbqxMpu97Z-TOEU8_Uyn_W3eH/view?usp=sharing)
    - [label](https://drive.google.com/file/d/1RBxLHHTf3CbHcuKFOybs-ILIuEJDiiBh/view?usp=sharing)
    - [data test 1](https://drive.google.com/file/d/1vpMqb4Cug3iKce-KzdadUi8x0HuckA06/view?usp=sharing)
    - [data test 2](https://drive.google.com/file/d/1m9QvLTw68b2wKWPr4p_g9b9O4VNz98XF/view?usp=sharing) hoặc [tại đây](https://drive.google.com/drive/folders/1toURLg1PsKv54bVyBFpfffmEFpnHjgno)

- B4. unzip data
    ```sh
    unzip "../data_zalo-challenge.zip" -d "zalo/dataset/train"
    unzip ".../label.zip" -d "zalo/dataset/train"
    unzip "/content/drive/MyDrive/dataAI/zalo/test.zip"
    unzip "../test2.zip" -d "zalo/dataset/test"
    ```
- B5. Train
    ```sh
    python zalo/train.py --name_model resnet50 --epochs 5 --lr 0.005 --batch_size 8 --replicate 11
    ```
    - model : ['resnet50', 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'] ( đang update )
    - Nếu bạn muốn train model riêng hãy thay tên và ở phần  parser.add_argument('--name_model'...) hãy thêm tên model vào phần choices
    - relicate : số frame crop từ 1 video, nếu muốn 1 số khác bạn cần download video về và để video tại : zalo/dataset.
        ```
        eg: zalo/dataset/public/videos
        ```
- B6. Test
    - data public 1 :
        ```sh
        python zalo/predict.py --public public --name_model resnet50 --replicate 11
        ```
    - data public 2 :
        ```sh
        python zalo/predict.py --public public2 --name_model resnet50
        ```  
    - hoặc
        ```sh
        bash predict.sh
        ```