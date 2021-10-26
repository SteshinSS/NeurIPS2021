def update_model_config(model_config: dict, preprocessed_data: dict):
    model_config['first_dim'].insert(0, preprocessed_data['first_features'])
    model_config['second_dim'].insert(0, preprocessed_data['second_features'])
    return model_config