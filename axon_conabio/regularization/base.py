import tensorflow as tf


class RegularizerBase(object):
    def __init__(self, config):
        self.config = config

    def filter_regularization_variables(self, variable):
        if 'bias' in variable.name:
            return False
        if not variable._trainable:
            return False
        return True

    def get_loss(self, model):
        pass


class DefaultRegularizer(RegularizerBase):
    def get_loss(self, model):
        reg_conf = self.config
        l1_loss = reg_conf['l1_loss']
        l2_loss = reg_conf['l2_loss']

        with model.graph.as_default():
            loss = 0
            if (l1_loss > 0 or l2_loss > 0):
                variables = [
                    variable for variable in model.variables.values()
                    if self.filter_regularization_variables(variable)]
                for var in variables:
                    if l1_loss > 0:
                        loss += tf.reduce_sum(tf.abs(var)) * l1_loss

                    if l2_loss > 0:
                        loss += tf.nn.l2_loss(var) * l2_loss
        return loss
