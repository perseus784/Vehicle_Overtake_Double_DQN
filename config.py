import os
import tensorflow as tf
batch_size = 32
n_vehicles = 20
node_history_size = 15000
generations = 3000
no_of_actions = 5
discount_factor = 0.99
save_path = 'files/training/model_files/cp-{}.ckpt'
save_dir = os.path.dirname(save_path)
log_save_path = os.path.join("files","training","my_logs")
if not os.path.exists(log_save_path):
    os.makedirs(log_save_path)
tf_writer = tf.summary.create_file_writer("files/training/my_logs/tf_board")

IM_W, IM_H = 200, 100


'''
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y"],
        "absolute": False,
        "order": "sorted",
        "normalize": True
    }
}
env.configure(config)'''
