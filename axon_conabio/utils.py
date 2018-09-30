from contextlib import contextmanager

import tensorflow as tf


TF_DTYPES = {
    'float': tf.float32,
    'float16': tf.float16,
    'float32': tf.float32,
    'float64': tf.float64,
    'bfloat16': tf.bfloat16,
    'complex': tf.complex64,
    'complex64': tf.complex64,
    'complex128': tf.complex128,
    'int': tf.int32,
    'int8': tf.int8,
    'uint8': tf.uint8,
    'uint16': tf.uint16,
    'uint32': tf.uint32,
    'uint64': tf.uint64,
    'int16': tf.int16,
    'int32': tf.int32,
    'int64': tf.int64,
    'bool': tf.bool,
    'string': tf.string,
    'qint8': tf.qint8,
    'quint8': tf.quint8,
    'qint16': tf.qint16,
    'quint16': tf.quint16,
    'qint32': tf.qint32,
}


def collection_scope(func, storage):
    def wrapper(*args, **kwargs):
        # All summary operations have name of summary as first argument. Should
        # this change, this might brake the code.
        name = args[0]
        func_name = func.__name__
        key = (name, func_name)
        if key not in storage:
            storage[key] = []
        storage[key].append((func, args, kwargs))

    return wrapper


@contextmanager
def summaries_scope(storage):
    old_image_summary = tf.summary.image
    old_audio_summary = tf.summary.audio
    old_histogram_summary = tf.summary.histogram
    old_scalar_summary = tf.summary.scalar
    old_tensor_summary_summary = tf.summary.tensor_summary

    # Replace tensorflow summary operations with custom decorator
    tf.summary.image = collection_scope(
            tf.summary.image,
            storage)
    tf.summary.scalar = collection_scope(
            tf.summary.scalar,
            storage)
    tf.summary.audio = collection_scope(
            tf.summary.audio,
            storage)
    tf.summary.histogram = collection_scope(
            tf.summary.histogram,
            storage)
    tf.summary.tensor_summary = collection_scope(
            tf.summary.tensor_summary,
            storage)

    yield

    # Restore tensorflow functions
    tf.summary.image = old_image_summary
    tf.summary.tensor_summary = old_tensor_summary_summary
    tf.summary.scalar = old_scalar_summary
    tf.summary.histogram = old_histogram_summary
    tf.summary.audio = old_audio_summary
