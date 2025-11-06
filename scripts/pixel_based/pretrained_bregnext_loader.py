import tensorflow as tf


def load_bregnext_model(checkpoint_path):
    """
    Load the pre-trained BReG-NeXt TensorFlow model from checkpoint.

    Args:
        checkpoint_path (str): Path to the TensorFlow checkpoint directory.

    Returns:
        tuple: A tuple containing the loaded TensorFlow session, input placeholder, and predictions tensor.
    """
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph(checkpoint_path + 'checkpoints-4300.meta')
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint(checkpoint_path))

    graph = tf.compat.v1.get_default_graph()
    new_input = graph.get_operation_by_name('image_batch_placeholder').outputs[0]
    predictions = graph.get_operation_by_name('FullyConnected/BiasAdd').outputs[0]

    return sess, new_input, predictions
