# Necessary packages
from tensorflow import keras
from tensorflow.keras import Model, Input, layers
from vime_utils import mask_generator, pretext_generator


def vime_self (x_unlab, p_m, alpha, parameters):
  """Self-supervised learning part in VIME.
  
  Args:
    x_unlab: unlabeled feature
    p_m: corruption probability
    alpha: hyper-parameter to control the weights of feature and mask losses
    parameters: epochs, batch_size
    
  Returns:
    encoder: Representation learning block
  """
    
  # Parameters
  _, dim = x_unlab.shape
  epochs = parameters['epochs']
  batch_size = parameters['batch_size']
  
  # Build model using Functional API
  inputs = Input(shape=(dim,))
  # Encoder
  h = layers.Dense(dim, activation='relu')(inputs)
  # Mask estimator
  mask_output = layers.Dense(dim, activation='sigmoid', name='mask')(h)
  # Feature estimator
  feature_output = layers.Dense(dim, activation='sigmoid', name='feature')(h)
  
  #Create model
  model = Model(inputs = inputs, outputs = [mask_output, feature_output])
  
  model.compile(optimizer='rmsprop',
                loss={'mask': 'binary_crossentropy', 
                      'feature': 'mean_squared_error'},
                loss_weights={'mask':1.0, 'feature':float(alpha)})
  
  # Generate corrupted samples
  m_unlab = mask_generator(p_m, x_unlab)
  m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
  
  # Fit model on unlabeled data
  model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab}, 
            epochs = epochs, batch_size= batch_size)
      
  # Extract encoder
  encoder = Model(
      inputs=model.input,
      outputs=model.layers[1].output,
      name='encoder'
  )
  
  return encoder