import joblib

model = joblib.load("models/pipeline.joblib")

print(model)

print("\nPipeline steps:")
for name, step in model.named_steps.items():
    print(f"{name}: {step}")

print("\nNumber of input features:", model.n_features_in_)
