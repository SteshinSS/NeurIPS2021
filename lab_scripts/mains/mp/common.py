def update_model_config(model_config: dict, preprocessed_data: dict):
    model_config['feature_extractor_dims'].insert(0, preprocessed_data['first_features'])
    model_config['regression_dims'].insert(0, model_config['feature_extractor_dims'][-1])
    model_config['regression_dims'].append(preprocessed_data['second_features'])
    model_config["total_correction_batches"] = preprocessed_data['total_correction_batches']
    model_config["batch_weights"] = preprocessed_data["train_batch_weights"]
    return model_config