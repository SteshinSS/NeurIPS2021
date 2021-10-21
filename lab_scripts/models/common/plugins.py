from torch import nn


def get_activation(activation_name: str):
    if activation_name == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_name == "tanh":
        return nn.Tanh()
    elif activation_name == "selu":
        return nn.SELU()
    elif activation_name == "relu":
        return nn.ReLU()


def selu_init(layer):
    if not isinstance(layer, nn.Linear):
        return
    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="linear")


def relu_init(layer):
    if not isinstance(layer, nn.Linear):
        return
    nn.init.orthogonal_(layer.weight)


class BiasSetter:
    def __init__(self, bias):
        self.bias = bias

    def __call__(self, layer):
        if not isinstance(layer, nn.Linear):
            return
        nn.init.constant_(layer.bias, self.bias)


def init(net, activation, bias=0.0):
    bias_setter = BiasSetter(bias)
    if activation == "selu":
        net.apply(selu_init)
    elif activation in ["relu", 'leaky_relu']:
        net.apply(relu_init)
    net.apply(bias_setter)
