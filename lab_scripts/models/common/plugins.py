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
    nn.init.constant_(layer.bias, 0)


def relu_init(layer):
    if not isinstance(layer, nn.Linear):
        return
    nn.init.orthogonal_(layer.weight)
    nn.init.constant_(layer.bias, 0)


def init(net, activation):
    if activation == "selu":
        net.apply(selu_init)
    elif activation == "relu":
        net.apply(relu_init)
