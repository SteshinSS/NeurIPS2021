import torch

def update_model_config(model_config: dict, preprocessed_data: dict):
    model_config['feature_extractor_dims'].insert(0, preprocessed_data['first_input_features'])
    model_config['regression_dims'].insert(0, model_config['feature_extractor_dims'][-1])
    model_config['regression_dims'].append(preprocessed_data['second_input_features'])
    if model_config['black_and_white']:
        model_config['total_batches'] = 2
    else:
        if model_config['inject_test']:
            all_batches = torch.cat([preprocessed_data['train_batch_idx'], preprocessed_data['test_batch_idx']], dim=0)
            model_config['total_batches'] = torch.unique(all_batches).shape[0]
        else:
            model_config["total_batches"] = torch.unique(
                preprocessed_data["train_batch_idx"]
            ).shape[0]
    model_config["batch_weights"] = preprocessed_data["train_batch_weights"]
    print(model_config['batch_weights'])
    return model_config