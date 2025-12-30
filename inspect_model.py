import joblib, pprint
m = joblib.load('heart_model.pkl')
print('TYPE:', type(m))
print('HAS named_steps:', hasattr(m, 'named_steps'))
if hasattr(m, 'named_steps'):
    print('PIPELINE STEPS:')
    pprint.pprint(list(m.named_steps.keys()))
else:
    print('SAMPLE ATTRS:')
    pprint.pprint([a for a in dir(m) if not a.startswith('_')][:200])
print('\nmodel_n_features_in_ =', getattr(m, 'n_features_in_', None))
print('model_feature_names_in_ =')
pprint.pprint(getattr(m, 'feature_names_in_', None))
