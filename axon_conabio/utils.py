from contextlib import contextmanager
import os

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
def summary_scope(storage):
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


def summary_aggregation(summaries, prefix=None):
    if prefix is None:
        prefix = ''

    summaries_list = []

    for key in summaries:
        name, func_name = key
        name = prefix + name

        tensors = []

        # Since all summaries are of the same type and have the
        # same name, it can be assumed that all summaries have been
        # called with the same arguments, hence we can take wlog
        # the first arguments.
        summary_function = summaries[key][0][0]
        global_arguments = summaries[key][0][1][2:]
        global_kwargs = summaries[key][0][2]

        for func, args, kwargs in summaries[key]:
            # Tensor argument is always the second argument in
            # tensorflow summary functions. Should this change this
            # part of the code may break.
            tensor = args[1]
            tensors.append(tensor)

        # Aggregation is different in the scalar case since it is
        # assumed to be a tensor containing a single value.
        if func_name == 'scalar':
            ntensors = len(tensors)
            aggregated_tensors = tf.add_n(tensors) / ntensors
        else:
            aggregated_tensors = tf.concat(tensors, axis=0)

        # Add first two arguments to arguments list
        global_arguments = (
                [name, aggregated_tensors] +
                list(global_arguments))

        summary = summary_function(
                *global_arguments,
                **global_kwargs)
        summaries_list.append(summary)

    summary_op = tf.summary.merge(summaries_list)
    return summary_op


def get_checkpoints(
        summary_dir,
        tf_subdir='tensorflow',
        npy_subdir='numpy'):
    tf_step = None
    npy_step = None

    tf_dir = os.path.join(summary_dir, tf_subdir)
    npy_dir = os.path.join(summary_dir, npy_subdir)

    tf_ckpt = tf.train.latest_checkpoint(tf_dir)
    if tf_ckpt is not None:
        tf_step = int(tf_ckpt.split('-')[-1])

    npy_ckpts = [
        x for x in os.listdir(npy_dir)
        if x[-4:] == '.npz']
    if len(npy_ckpts) > 0:
        npy_ckpts = sorted(
            npy_ckpts,
            key=lambda x: int(x.split('.')[0].split('_')[-1]))
        npy_ckpt = os.path.join(npy_dir, npy_ckpts[-1])
        npy_step = int(npy_ckpt.split('.')[0].split('_')[-1])

    if (npy_step is None and tf_step is None):
        return None

    if npy_step is None:
        npy_step = -1

    if tf_step is None:
        tf_step = -1

    if npy_step >= tf_step:
        return 'numpy', npy_ckpt, npy_step
    else:
        return 'tf', tf_ckpt, tf_step
