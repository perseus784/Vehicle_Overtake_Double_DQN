import tensorflow as tf
import numpy as np
from config import *
import random
import os
import datetime
import csv 

class DeepQnetwork:

    def __init__(self, training_mode):
        self.epsilon, self.min_epsilon = 1, 0.1
        self.decay = self.epsilon/((generations//2)-1)
        self.training_mode = training_mode
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
        self.train_network = self.build_network()
        self.predict_network = self.build_network()
        self.counter=0

    '''def get_data(self):
        random_choice=random.randrange(len(input_val))
        return np.array([input_val[random_choice]]),np.array([one_hot_op[random_choice]])'''

    def record_summary(self, _loss, counter):
        summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=_loss)])
        self.writer.add_summary(summary, counter)

    '''def save_model(self):
        self.saver.save(self.sess, os.path.join(*[".","checkpoints","rl_weights"]),global_step=1000)'''

    def load_model(self):
        latest_weights = tf.train.latest_checkpoint(save_dir)
        self.predict_network.load_weights(latest_weights)

    def update_prediction_network(self):
        for train_grad, pred_grad in zip(self.train_network.trainable_variables, self.train_network.trainable_variables):
            pred_grad.assign(train_grad)
        self.train_network.save_weights(save_path.format(self.counter))
        print("leveling up")

    def update_q_value(self, rewards, current_q_list, next_q_list, actions, done):
        current_q_list = current_q_list.numpy()
        next_max_qs = np.max(next_q_list, axis=1)
        new_qs = rewards + (np.ones(done.shape)-done)*discount_factor * next_max_qs
        for i in range(len(current_q_list)):
            current_q_list[i, actions[i]] = new_qs[i]
        return current_q_list

    def loss(self, ground_truth, prediction):
        loss = tf.keras.losses.mean_squared_error(ground_truth, prediction)
        return loss

    def get_action(self, state):
        if np.random.random() > self.epsilon:
            _action = self.get_prediction(np.expand_dims(state, axis=0))
            action = np.argmax(_action)
        else:
            action = np.random.randint(0, no_of_actions)
        return action

    def get_prediction(self, states):
        states = np.reshape(states, newshape=(states.shape[0], IM_H, IM_W, 4))/255
        prediction = self.predict_network(states)
        return prediction

    def predict(self, states):
        states = np.reshape(states, newshape=(states.shape[0], IM_H, IM_W, 4))/255
        prediction = self.train_network(states)
        return prediction

    def save_log(self, step, quantity, filename):
        with open(os.path.join(log_save_path, filename), 'a+') as fi:
            csv_w = csv.writer(fi, delimiter=',')
            csv_w.writerow([step, quantity])

    @tf.function
    def train_step(self, states, actions):
        with tf.GradientTape() as tape:
            predictions = self.train_network(states)
            loss = self.loss(actions, predictions)
        gradients = tape.gradient(loss, self.train_network.trainable_variables)
        gradients = [tf.clip_by_norm(gradient, 10) for gradient in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.train_network.trainable_variables))
        return loss

    def build_network(self):

        '''inp = tf.keras.layers.Input((5,5)) 
        x = tf.keras.layers.Flatten()(inp)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(no_of_actions, activation='linear')(x)'''    

        inp = tf.keras.layers.Input((IM_H, IM_W, 4))
        x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inp)
        x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(x)
        x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
        x = tf.keras.layers.MaxPool2D((2,2))(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
        x = tf.keras.layers.MaxPool2D((2,2))(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
        x = tf.keras.layers.Conv2D(256, (3,3), activation='relu')(x)
        x = tf.keras.layers.MaxPool2D((2,2))(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Conv2D(512, (3,3), activation='relu')(x)
        x = tf.keras.layers.MaxPool2D((2,2))(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(no_of_actions, activation='linear')(x)

        model = tf.keras.Model(inputs=inp, outputs=x)
        model.summary()
        return model

    def train(self, previous_memories):
        self.counter += 1
        current_nodes, actions, next_nodes, rewards, done = previous_memories
        #print(current_nodes.shape, actions.shape, rewards.shape, next_nodes.shape)
        current_action_qs = self.predict(current_nodes)
        next_action_qs = self.get_prediction(next_nodes)
        current_action_qs = self.update_q_value(rewards, current_action_qs, next_action_qs, actions, done)
        current_nodes = np.reshape(current_nodes, newshape=(batch_size, IM_H, IM_W, 4))/255

        loss = self.train_step(current_nodes, current_action_qs)
        
        with tf_writer.as_default():
            tf.summary.scalar("loss", data=np.mean(loss), step=self.counter)

        self.save_log(self.counter, np.mean(loss), "loss.csv")

