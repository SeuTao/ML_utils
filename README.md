# ML Toolbox

# Examples
## Train scikit-learn API model
- CatBoost
```
CAT_PARMAS = {
    'iterations': 1000, 'depth': 4
}
CAT_FIT_PARAMS = {
    'early_stopping_rounds': 100, 'plot': False
}

model = Trainer(CatBoostClassifier(**CAT_PARAMS))
model.train(x_train, y_train, x_valid, y_valid, fit_params=CAT_FIT_PARAMS)
```

## Cross validate scikit-learn API model
- Logistic regression
```
LOGI_PARAMS = {
    'C': 1, 'class_weight': 'balanced'
}

skf = StratifiedKFold(n_splits=CV, shuffle=True)
logi_cv = CrossValidator(LogisticRegression(**LOGI_PARAMS), skf)
logi_cv.run(
    X, y, x_test, 
    eval_metric=[AUC(), SeUnderSp(sp=0.9)], 
    pred_method='binary_proba_positive', 
    transform=InfoldTargetEncoder(),
    importance_method='permutation',
    verbose=0
)
logi_cv.plot_feature_importances()
```

- LightGBM
```
LGB_PARAMS = {
    'num_leaves': 8,
    'objective': 'binary', 'num_tree': 2000, 
    'metric':'auc', 'learning_rate': 0.03, 'boosting_type': 'gbdt'
}
LGB_FIT_PARAMS = {
    'verbose': 10, 'early_stopping_rounds': 100, 
    'eval_metric': 'auc'
}

skf = StratifiedKFold(n_splits=CV, shuffle=True)
lgb_cv = CrossValidator(LGBMClassifier(**LGB_PARAMS), skf)
lgb_cv.run(
    X, y, x_test, 
    eval_metric=[AUC(), SeUnderSp(sp=0.9)],
    pred_method='binary_proba', 
    cat_features=CAT_IDXS, 
    fit_params=LGB_FIT_PARAMS,
    verbose=0
)
lgb_cv.plot_feature_importances()
```

## Train PyTorch Model
```
model = TabularNet(X.shape[1], 1)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)
NN_FIT_PARAMS = {
    'loader': loader_train,
    'loader_valid': loader_valid,
    'loader_test': loader_test,
    'criterion': nn.BCEWithLogitsLoss(),
    'optimizer': optimizer,
    'scheduler': StepLR(optimizer, step_size=5, gamma=0.9),
    'num_epochs': 100, 
    'stopper': EarlyStopping(patience=20, maximize=True),
    'logger': Logger('results/test/'), 
    'snapshot_path': Path('results/test/nn_best.pt'),
    'eval_metric': AUC().torch,
    'info_format': '[epoch] time data loss metric earlystopping',
    'info_train': True,
    'info_interval': 3
}
trainer = TorchTrainer(model, serial='test')
trainer.fit(**NN_FIT_PARAMS)
```

## Cross validate PyTorch Model for tabular data
```
model = TabularNet(X.shape[1], 1)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)
skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=SEED)
NN_FIT_PARAMS = {
    'criterion': nn.BCEWithLogitsLoss(),
    'optimizer': optimizer,
    'scheduler': StepLR(optimizer, step_size=10, gamma=0.9),
    'num_epochs': 50, 
    'calibrate_model': True, 
    'stopper': EarlyStopping(patience=5, maximize=True),
    'eval_metric': AUC().torch,
    'info_format': '[epoch] time data loss metric earlystopping',
    'info_train': False,
    'info_interval': 5,
    'verbose': False
}

dnn_cv = TorchCV(model, skf)
dnn_cv.run(
    X, y, x_test, task='binary', 
    eval_metric=[AUC(), Accuracy(), SeUnderSp(sp=0.9)], 
    batch_size=2048,
    snapshot_dir='results/test/',
    fit_params=NN_FIT_PARAMS, 
    logger=Logger('results/test/'), 
    transform=InfoldTargetEncoder(CAT_IDXS, 
        encoder=ce.TargetEncoder(cols=np.arange(len(CAT_IDXS)), return_df=False))
)
```