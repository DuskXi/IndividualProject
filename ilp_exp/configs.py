from torch import nn

exp_attr_base = {
    'hidden_size': 256,
    'layers': 3,
    'activation_function': nn.ReLU,
    'layer_norm': True,
    'dropout': 0.1,
    'name': 'base_config'
}
exp_attr = [
    {
        'hidden_size': 128,
        'layers': 3,
        'activation_function': nn.ReLU,
        'layer_norm': True,
        'dropout': 0.1,
        'name': 'base_config_hidden_size_128'
    },
    {
        'hidden_size': 256,
        'layers': 3,
        'activation_function': nn.ReLU,
        'layer_norm': True,
        'dropout': 0.1,
        'name': 'base_config'
    },
    {
        'hidden_size': 512,
        'layers': 3,
        'activation_function': nn.ReLU,
        'layer_norm': True,
        'dropout': 0.1,
        'name': 'base_config_hidden_size_512'
    },
    {
        'hidden_size': 256,
        'layers': 6,
        'activation_function': nn.ReLU,
        'layer_norm': True,
        'dropout': 0.1,
        'name': 'base_config_layers_6'
    },
    {
        'hidden_size': 256,
        'layers': 6,
        'activation_function': nn.ReLU,
        'layer_norm': True,
        'dropout': 0.1,
        'name': 'base_config_layers_8'
    },
    {
        'hidden_size': 256,
        'layers': 3,
        'activation_function': nn.ReLU,
        'layer_norm': True,
        'dropout': 0,
        'name': 'base_config_no_dropout'
    },
    {
        'hidden_size': 256,
        'layers': 3,
        'activation_function': nn.GELU,
        'layer_norm': True,
        'dropout': 0.1,
        'name': 'base_config_gelu'
    },
    {
        'hidden_size': 256,
        'layers': 3,
        'activation_function': nn.LeakyReLU,
        'layer_norm': True,
        'dropout': 0.1,
        'name': 'base_config_leaky_relu'
    },
]
