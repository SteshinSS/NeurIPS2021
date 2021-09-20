from lab_scripts.models.baselines import official

def predict(input_train_mod1, input_train_mod2, input_test_mod1):
    return official.fit_predict(input_train_mod1, input_train_mod2, input_test_mod1)

if __name__=='__main__':
    predict()