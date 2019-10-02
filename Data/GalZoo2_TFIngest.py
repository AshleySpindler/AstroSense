import tensorflow as tf

def _parse_function(proto):
    #define the tfrecord, image as string
    keys_to_features = {'image': tf.FixedLenFeature([], tf.string)}

    #load an example
    parsed_features = tf.parse_single_example(proto, keys_to_features)

    #turn into array
    parsed_features['image'] = tf.decode_raw(parsed_features['image'], tf.uint8)

    return parsed_features['image']

def create_dataset(filepath, batch_size, shuffle_size):

    dataset = tf.data.TFRecordDataset(filepath)

    dataset = dataset.map(_parse_function)

    #dataset = dataset.shuffle(shuffle_size)

    dataset = dataset.batch(batch_size)

    dataset = dataset.repeat(-1)

    iterator = tf.data.make_one_shot_iterator(dataset)

    image = iterator.get_next()

    image = tf.reshape(image, [-1, 64, 64, 1])

    image = tf.divide(image, 255)

    return image
