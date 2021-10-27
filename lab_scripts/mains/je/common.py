def update_model_config(model_config: dict, preprocessed_data: dict):
    model_config['first_dim'].insert(0, preprocessed_data['first_features'])
    model_config['second_dim'].insert(0, preprocessed_data['second_features'])
    model_config['common_dim'].insert(0, model_config['first_dim'][-1] + model_config['second_dim'][-1])
    return model_config