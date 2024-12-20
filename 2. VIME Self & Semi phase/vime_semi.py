# Necessary packages
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow import keras
from vime_utils import mask_generator, pretext_generator
from sklearn.metrics import cohen_kappa_score

 
"""
Expected flow:
- Load & process data
- Create network architecture
- Train the model using:
+ Supervised loss from labeled data
+ Unsupervised loss from unlabeled data
"""

class Predictor(keras.Model):
    def __init__(self, hidden_dim, label_dim):
        super(Predictor, self).__init__()
        #Define layers
        self.dense1 = keras.layers.Dense(hidden_dim, activation='relu')
        self.dense2 = keras.layers.Dense(hidden_dim, activation='relu')
        self.output_layer = keras.layers.Dense(label_dim)
        
    def call(self, x_input):
        #Forward pass
        inter_layer = self.dense1(x_input)
        inter_layer = self.dense2(inter_layer)
        y_hat_logit = self.output_layer(inter_layer)
        y_hat = tf.nn.softmax(y_hat_logit)
        return y_hat_logit, y_hat
 
def vime_semi(x_train, y_train, x_unlab, x_test, parameters, 
              p_m, K, beta, file_name):
    """Semi-supervied learning part in VIME.
    
    Args:
        - x_train, y_train: training dataset
        - x_unlab: unlabeled dataset
        - x_test: testing features
        - parameters: network parameters (hidden_dim, batch_size, iterations)
        - p_m: corruption probability
        - K: number of augmented samples
        - beta: hyperparameter to control supervised and unsupervised loss
        - file_name: saved filed name for the encoder function
        
    Returns:
        - y_test_hat: prediction on x_test
    """
    class WeightedKappaLoss(tf.keras.losses.Loss):
        def __init__(self, num_classes, name="weighted_kappa_loss"):
            super().__init__(name=name)
            self.num_classes = num_classes
            # Create weight matrix
            weights = tf.cast(tf.range(num_classes), tf.float32)
            weights = tf.expand_dims(weights, 0) - tf.expand_dims(weights, 1)
            self.weights = tf.square(weights)
            
        @tf.function
        def call(self, y_true, y_pred):
            # Convert types
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            # Apply softmax
            y_pred = tf.nn.softmax(y_pred)
            
            # Calculate confusion matrix
            batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
            confusion = tf.matmul(y_true, y_pred, transpose_a=True)
            confusion = confusion / batch_size
            
            # Calculate agreements
            observed = tf.reduce_sum(confusion * (1.0 - self.weights))
            expected_rows = tf.reduce_sum(confusion, axis=1)
            expected_cols = tf.reduce_sum(confusion, axis=0)
            expected = tf.reduce_sum(
                tf.matmul(tf.expand_dims(expected_rows, 1),
                         tf.expand_dims(expected_cols, 0)) * (1.0 - self.weights)
            )
            
            # Calculate kappa
            kappa = (observed - expected) / (1.0 - expected + 1e-8)
            return 1.0 - kappa
    
    # Network parameters
    hidden_dim = parameters['hidden_dim']
    batch_size = parameters['batch_size']
    iterations = parameters['iterations']
      
    # Basic parameters
    data_dim = x_train.shape[1]
    label_dim = y_train.shape[1]

    # Divide training and validation sets (9:1)
    idx = np.random.permutation(len(x_train))
    train_idx = idx[:int(len(idx)*0.9)]
    valid_idx = idx[int(len(idx)*0.9):]
    
    x_valid, y_valid = x_train[valid_idx], y_train[valid_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]
    
    # Load encoder from self-supervised model
    encoder = keras.models.load_model(file_name)
    
    # Create predictor model
    predictor = Predictor(hidden_dim, label_dim)
    optimizer = keras.optimizers.Adam()
    
    # Encode validation and testing features
    x_valid = encoder.predict(x_valid)
    x_test = encoder.predict(x_test)
    
    # Setup checkpointing
    checkpoint = tf.train.Checkpoint(model=predictor)
    manager = tf.train.CheckpointManager(
        checkpoint, './save_model', max_to_keep=1)
    
    # Early stopping variables
    best_valid_loss = float('inf')
    patience = -1
    
    # Initialize loss
    loss_fn = WeightedKappaLoss(num_classes=label_dim)
    
    @tf.function
    def train_step(x_batch, y_batch, xu_batch):
        with tf.GradientTape() as tape:
            #Forward pass
            y_logits, _ = predictor(x_batch)
            yv_logits, _ = predictor(xu_batch)
            
            # Calculate supervised loss
            supervised_loss = loss_fn(y_batch, y_logits)

            # Unsupervised loss using variance
            unsupervised_loss = tf.reduce_mean(
                tf.math.reduce_variance(yv_logits, axis=0))
            
            total_loss = supervised_loss + beta * unsupervised_loss
        
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, predictor.trainable_variables)
        optimizer.apply_gradients(zip(gradients, predictor.trainable_variables))
        
        return total_loss
        
    for it in range(iterations):
        # Select a batch of labeled data
        batch_idx = np.random.permutation(len(x_train))[:batch_size]
        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]
        
        # Encode labeled data
        x_batch = encoder.predict(x_batch)
        
        # Select and augment unlabeled data
        batch_u_idx = np.random.permutation(len(x_unlab))[:batch_size]
        xu_batch_ori = x_unlab[batch_u_idx]
        
        xu_batch = []
        for _ in range(K):
            # Mask vector generation
            m_batch = mask_generator(p_m, xu_batch_ori)
            # Pretext generator
            _, xu_batch_temp = pretext_generator(m_batch, xu_batch_ori)
            
            # Encode corrupted samples
            xu_batch_temp = encoder.predict(xu_batch_temp)
            xu_batch.append(xu_batch_temp)
        
        #Convert list to matrix
        xu_batch = tf.convert_to_tensor(np.array(xu_batch))
        
        # Training step
        loss = train_step(x_batch, y_batch, xu_batch)
        
        # Validation step
        val_logits, _ = predictor(x_valid)
        
        val_loss = loss_fn(y_valid, val_logits)
        
        if it % 100 == 0:
            print(f'Iteration: {it}/{iterations}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
            
        # Early stopping
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            manager.save()
            patience = 0
        else:
            patience += 1
            if patience >= 100:
                break
            
        
    # Restore best model
    checkpoint.restore(manager.latest_checkpoint)
    
    # Generate predictions
    _, y_test_hat = predictor(x_test)
    
    return y_test_hat.numpy()