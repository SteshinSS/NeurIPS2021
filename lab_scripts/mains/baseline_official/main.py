from lab_scripts.models.baselines import baseline_official

def main(input_train_mod1, input_train_mod2, input_test_mod1):
    return baseline_official.fit_predict(input_train_mod1, input_train_mod2, input_test_mod1)