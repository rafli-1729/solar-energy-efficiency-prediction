import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_importance(pipeline, X, name, top_n=25, palette="viridis", save_path=None):
    wrapper = pipeline.named_steps['model']
    if hasattr(wrapper, 'regressor_'):
        model = wrapper.regressor_
    else:
        model = wrapper

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'booster_'):
        importances = model.booster_.feature_importance(importance_type='gain')
    else:
        raise ValueError("Model does not have feature importances feature")

    fe_step = pipeline.named_steps['feature_engineering']
    feature_names = fe_step.get_feature_names_out()

    if len(feature_names) != len(importances):
        raise ValueError("Feature names and importances lenght does not match")

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(12, 7))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=df,
        palette=palette
    )
    plt.title(f"Top {top_n} Feature Importance of {name} Model", fontsize=16, weight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.tight_layout()
    plt.show()
