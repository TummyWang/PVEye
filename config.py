class Config:
    train_dir = r'D:\PVeye\train'
    test_dir = r'D:\PVeye\test_side'
    batch_size = 512
    test_batch_size = 3150
    epochs = 200
    gpu_id=0
    learning_rate = 0.0001
    momentum = 0.9
    eye_side = 'both'
    save_path = r'D:\PVeye\ckpt\\NVGaze_model_{epoch}.pth'
    load_model_path  = r"D:\PVeye\ckpt\NVGaze_model.pth"
    save_step=5

