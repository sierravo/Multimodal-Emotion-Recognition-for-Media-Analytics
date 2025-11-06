import tensorflow as tf

BREGNEXT_CHECKPOINT = 'checkpoints/categorical_attempt_3/'

def load_bregnext_model():
    """
    Load the BReG-NeXt model graph and return session, input placeholder, and prediction tensor.

    Returns:
        sess (tf.Session): TensorFlow session with loaded graph.
        input_tensor (tf.Tensor): Placeholder for image input.
        prediction_tensor (tf.Tensor): Output prediction tensor.
    """
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph(BREGNEXT_CHECKPOINT + 'checkpoints-4300.meta')
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint(BREGNEXT_CHECKPOINT))
    graph = tf.compat.v1.get_default_graph()
    input_tensor = graph.get_operation_by_name('image_batch_placeholder').outputs[0]
    prediction_tensor = graph.get_operation_by_name('FullyConnected/BiasAdd').outputs[0]
    return sess, input_tensor, prediction_tensor
