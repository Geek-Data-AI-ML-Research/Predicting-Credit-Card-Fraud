
# %% cell 5
CURRENT_CFG = {'recipe_key': 'h10_small_trees',
 'view': 'digits',
 'model_params': {'n_estimators': 1600,
                  'num_leaves': 4,
                  'learning_rate': 0.03,
                  'random_state': 42,
                  'n_jobs': 4,
                  'min_child_samples': 100,
                  'reg_lambda': 1.0,
                  'subsample': 0.8,
                  'subsample_freq': 1,
                  'colsample_bytree': 0.5,
                  'verbosity': -1,
                  'deterministic': True,
                  'force_col_wise': True},
 'drop_predictors': [],
 'n_splits': 5,
 'split_seed': 42,
 'export_mode': 'cv_average'}

# %% cell 7
CFG = {'competition': 'playground-series-s6e11',
 'title': 'S6E11 Fraud | LB 0.95095 | Shallow Digit Models',
 'confirmed': True,
 'task': 'binary',
 'metric': 'auc',
 'target': 'is_fraud',
 'id_col': 'id',
 'positive_class': 1,
 'class_columns': None,
 'split': 'iid',
 'seed': 42,
 'n_splits': 5,
 'n_estimators': 300,
 'num_leaves': 31,
 'learning_rate': 0.05,
 'n_jobs': 4,
 'drop_features': [],
 'category_view': 'auth_method',
 'run_research': True,
 'run_repeated_frequency': False,
 'run_repeated_h4': False,
 'run_candidate_studies': False,
 'run_h5_diagnosis': False,
 'run_overlap_probe': False,
 'run_prospective_audit': False,
 'run_current_recipe': True,
 'run_digit_screen': False,
 'run_digit_audit': False,
 'run_representation_study': False,
 'run_shallow_refit_study': False,
 'run_seed_bag': False,
 'run_glm_study': False}

# %% cell 8
import time
notebook_started = time.perf_counter()
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from IPython import get_ipython
_shell = get_ipython()
if _shell is not None and hasattr(_shell, "kernel"):
    _shell.run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
from IPython.display import display

plt.rcParams.update({"figure.figsize": (7.4, 3.8), "figure.dpi": 98,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

def locate_data():
    override = os.environ.get("LOCAL_DATA")
    candidates = [Path(override)] if override else [
        Path("/kaggle/input/competitions") / CFG["competition"],
        Path("/kaggle/input") / CFG["competition"], Path("data")]
    for path in candidates:
        if all((path / name).is_file() for name in ("train.csv", "test.csv", "sample_submission.csv")):
            return path
    return None

DATA = locate_data()
ready = DATA is not None and CFG["confirmed"]
if not ready:
    raise FileNotFoundError("Attach the S6E11 competition files and set CFG['confirmed']=True. "
                            "For a local run, set LOCAL_DATA to their directory.")
else:
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.metrics import (roc_auc_score, log_loss, mean_squared_error,
                                 mean_absolute_error, accuracy_score, balanced_accuracy_score)
    from lightgbm import LGBMClassifier, LGBMRegressor
    task, metric = CFG["task"], CFG["metric"]
    allowed = {"binary": {"auc", "logloss", "accuracy", "balanced_accuracy"},
               "multiclass": {"logloss", "accuracy", "balanced_accuracy", "map@3"},
               "regression": {"rmse", "mae", "rmsle"}}
    if task not in allowed or metric not in allowed[task]:
        raise ValueError("Set a supported official task/metric pair in CFG; do not infer the metric.")
    if CFG["split"] != "iid":
        raise ValueError("Replace the split with a group/time-aware design before running this task.")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    id_col = CFG["id_col"] or sample.columns[0]
    targets = [c for c in train if c not in test]
    target = CFG["target"] or (targets[0] if len(targets) == 1 else None)
    if target is None or target not in train or target in test:
        raise ValueError(f"Confirm target explicitly; train-only columns: {targets}")
    if id_col not in train or id_col not in test or id_col not in sample:
        raise ValueError("Confirm the ID column in train, test and sample_submission.")
    for name, frame in [("train", train), ("test", test), ("sample", sample)]:
        if frame[id_col].isna().any() or not frame[id_col].is_unique:
            raise ValueError(f"{name} IDs must be nonmissing and unique.")
    if len(sample) != len(test) or set(sample[id_col]) != set(test[id_col]):
        raise ValueError("sample_submission and test must contain the same IDs.")
    if train[target].isna().any():
        raise ValueError("Target contains missing values.")
    schema_features = [c for c in train if c not in {id_col, target}]
    if not schema_features or set(schema_features) != set(test.columns) - {id_col}:
        raise ValueError("Train/test feature columns differ.")
    drop_features = CFG.get("drop_features", [])
    if set(drop_features) - set(schema_features):
        raise ValueError("drop_features contains an unknown column.")
    features = [c for c in schema_features if c not in drop_features]
    if not features:
        raise ValueError("At least one model feature is required.")
    categorical = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
    numeric = [c for c in features if c not in categorical]
    print(json.dumps({"competition": CFG["competition"], "task": task, "metric": metric,
                      "target": target, "train_rows": len(train), "test_rows": len(test),
                      "features": len(features)}, indent=2))
    print("Excluded model features:", drop_features or "none")

# %% cell 9
"""Presentation helpers only; experiment calculations stay in visible notebook cells."""
import base64
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython.display import HTML, display


COLOURS = {"Original": "#D55E00", "S6E11 train": "#0072B2"}


def setup_style():
    """Apply a light, colourblind-safe style for the notebook's figures."""
    plt.rcParams.update({
        "figure.figsize": (7.3, 4.5), "figure.dpi": 110,
        "savefig.dpi": 150, "font.family": "DejaVu Sans", "font.size": 14,
        "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlepad": 16,
        "axes.labelsize": 14, "axes.labelpad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B8C4CC", "axes.labelcolor": "#233441",
        "text.color": "#14212B", "xtick.color": "#52616E",
        "ytick.color": "#52616E", "xtick.labelsize": 14, "ytick.labelsize": 14,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "grid.color": "#E4EAF0", "grid.linewidth": .7, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 14,
        "axes.prop_cycle": plt.cycler(color=["#0072B2", "#D55E00", "#009E73", "#CC79A7"]),
    })


def styled_table(frame, formats=None):
    """Return a compact 13px Styler without changing any table values."""
    styler = frame.style
    if formats:
        styler = styler.format(formats, na_rep="n/a")
    return (styler.set_properties(**{
        "font-size": "13px", "color": "#233441", "padding": "7px 9px",
        "border-bottom": "1px solid #E4EAF0", "background-color": "white",
        "white-space": "nowrap",
    }).set_table_styles([
        {"selector": "", "props": [("display", "block"), ("max-width", "100%"),
                                     ("overflow-x", "auto"), ("font-family", "Arial, sans-serif"),
                                     ("font-size", "13px"), ("border-collapse", "collapse")]},
        {"selector": "th", "props": [("font-size", "13px"), ("background-color", "#F5F8FA"),
                                      ("color", "#233441"), ("padding", "7px 9px"),
                                      ("border-bottom", "1px solid #D5DEE5"),
                                      ("text-align", "left")]},
        {"selector": "caption", "props": [("font-size", "13px"), ("text-align", "left"),
                                           ("color", "#52616E"), ("padding", "6px 0")]},
    ]))


def callout(text):
    """Return the shared information-callout treatment; text is plain text."""
    body = escape(str(text)).replace("\n", "<br/>")
    return HTML('<div class="alert alert-info" style="background:#EEF6FB;'
                'border:1px solid #D3E7F4;border-left:4px solid #0072B2;'
                'border-radius:5px;padding:12px 15px;margin:15px 0;'
                'color:#233441;font-size:14px;line-height:1.55;">' + body + '</div>')


def source_hero(source_summary, amount_quantiles):
    """Plot supplied summary values and amount quantiles; return an unsaved Figure."""
    labels = ("Original", "S6E11 train")
    summary = source_summary.set_index("dataset").loc[list(labels)]
    amounts = amount_quantiles.sort_values("quantile")
    if (amounts[list(labels)] <= 0).any().any():
        raise ValueError("Logarithmic amount display requires positive supplied quantiles.")

    fig, (rates, ecdf) = plt.subplots(2, 1, figsize=(7.3, 8.6))
    fig.subplots_adjust(left=.16, right=.97, top=.86, bottom=.12, hspace=.80)
    # Larger text than the ordinary EDA figures: the first-screen hero also fits a phone.
    fig.suptitle("Original source and S6E11 train", fontsize=18, weight="bold", x=.565)
    values = summary["fraud_rate"].to_numpy() * 100
    rates.bar(labels, values, color=[COLOURS[label] for label in labels], width=.48)
    for position, label in enumerate(labels):
        row = summary.loc[label]
        rates.annotate(f'{row.fraud_rate:.2%}\n{int(row.fraud):,} / {int(row.rows):,}',
                       (position, values[position]), xytext=(0, 9), textcoords="offset points",
                       ha="center", va="bottom", fontsize=15, color="#233441", linespacing=1.35)
    rates.set_ylim(0, max(values.max() * 1.65, .1))
    rates.set_ylabel("Fraud share (%)", fontsize=15)
    rates.set_title("Label prevalence", fontsize=16, loc="left", pad=17)
    rates.tick_params(axis="both", labelsize=15)
    rates.grid(axis="y")
    rates.set_xlim(-.65, 1.65)

    for label, linestyle in zip(labels, ("--", "-")):
        ecdf.plot(amounts[label], amounts["quantile"], label=label,
                  color=COLOURS[label], linestyle=linestyle, linewidth=2.5)
    ecdf.set_xscale("log")
    ecdf.set_ylim(0, 1)
    ecdf.yaxis.set_major_formatter(PercentFormatter(1))
    ecdf.set_xlabel("Transaction amount (USD, log scale)", fontsize=15)
    ecdf.set_ylabel("Cumulative share", fontsize=15)
    ecdf.set_title("Transaction amounts", fontsize=16, loc="left", pad=43)
    ecdf.tick_params(axis="both", labelsize=15)
    ecdf.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=2, fontsize=14,
                borderaxespad=0, handlelength=2.2, columnspacing=1.4)
    ecdf.grid(axis="both", which="major")
    return fig


def show_figure(fig, caption, filename):
    """Save and display a figure, place its caption below, and release it."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    encoded = base64.b64encode(path.read_bytes()).decode()
    display(HTML('<img src="data:image/png;base64,' + encoded + '" width="730" '
                 'style="max-width:100%;height:auto;" alt="' + escape(str(caption), quote=True) + '"/>'))
    display(HTML('<p style="font-size:13px;line-height:1.5;color:#52616E;'
                 'margin:5px 0 18px;">' + escape(str(caption)) + '</p>'))
    plt.close(fig)
    return None

setup_style()

# %% cell 11
dictionary = [('Amounts', 'amount_usd; account_balance_usd', 'USD'),
 ('Timing and activity',
  'hours_since_last_txn; txn_count_last_24h; time_of_day_hour; day_of_week',
  'Hours, count, hour, coded day'),
 ('Customer and card',
  'distance_from_home_km; card_age_months; customer_age',
  'km, months, age'),
 ('Supplied scores and counts',
  'cvv_retry_count; velocity_score; merchant_risk_score; prior_disputes',
  'Counts or supplied scores'),
 ('Categories',
  'merchant_category; card_type; auth_method; channel; device_type',
  'Observed text levels'),
 ('Boolean flags',
  'is_foreign_transaction; is_new_merchant; used_vpn; ip_country_mismatch; '
  'billing_shipping_mismatch; is_ai_generated_scam_attempt',
  'False / True')]
display(styled_table(pd.DataFrame(dictionary, columns=["group", "columns", "units / coding"]).set_index("group")).set_properties(**{"white-space":"normal", "overflow-wrap":"anywhere"}).set_table_styles([{"selector":"th.row_heading", "props":[("white-space","normal"), ("max-width","130px")]}], overwrite=False))

# %% cell 14
from scipy.stats import ks_2samp
original_paths = [Path('original/credit_card_fraud_2026.csv'),
    Path('/kaggle/input/credit-card-fraud-detection-2026/credit_card_fraud_2026.csv'),
    Path('/kaggle/input/datasets/uditjain13/credit-card-fraud-detection-2026/credit_card_fraud_2026.csv')]
original_path = next((p for p in original_paths if p.is_file()), None)
if original_path is None:
    raise FileNotFoundError('Attach Credit Card Fraud Detection 2026 to reproduce the source comparison.')
original = pd.read_csv(original_path)
assert set(original[target].unique()) <= {0, 1}
assert set(schema_features) <= set(original.columns)
frames = {'Original': original, 'S6E11 train': train, 'S6E11 test': test}
source_summary = pd.DataFrame([
    dict(dataset=name, rows=len(frame), fraud=int(frame[target].sum()),
         fraud_rate=frame[target].mean(), median_amount=frame.amount_usd.median())
    for name, frame in list(frames.items())[:2]])
amount_quantiles = pd.DataFrame({'quantile': np.linspace(0, 1, 401)})
for name in ['Original', 'S6E11 train']:
    amount_quantiles[name] = frames[name].amount_usd.quantile(amount_quantiles['quantile']).to_numpy()
display(styled_table(source_summary.set_index('dataset'),
    {'rows': '{:,.0f}', 'fraud': '{:,.0f}', 'fraud_rate': '{:.3%}', 'median_amount': '${:.2f}'}))

# %% cell 18
# Audit every supplied predictor, even when a model ablation excludes one.
audit_numeric = [c for c in schema_features if pd.api.types.is_numeric_dtype(train[c])
                 and not pd.api.types.is_bool_dtype(train[c])]
audit_categories = [c for c in schema_features if c not in audit_numeric]
quality = pd.DataFrame({'type': train[schema_features].dtypes.astype(str),
    'missing train': train[schema_features].isna().sum(),
    'missing test': test[schema_features].isna().sum(),
    'distinct train': train[schema_features].nunique(),
    'distinct test': test[schema_features].nunique()})
train_hash = pd.util.hash_pandas_object(train[schema_features], index=False)
test_hash = pd.util.hash_pandas_object(test[schema_features], index=False)
checks = {'duplicate_feature_rows_train': int(train_hash.duplicated().sum()),
    'duplicate_feature_rows_test': int(test_hash.duplicated().sum()),
    'test_rows_matching_train_features': int(test_hash.isin(train_hash).sum()),
    'shared_train_test_ids': len(set(train[id_col]) & set(test[id_col])),
    'constant_train_features': [c for c in schema_features if train[c].nunique(dropna=False) == 1],
    'numeric_infinities_train': int(np.isinf(train[audit_numeric].to_numpy(float)).sum()),
    'numeric_infinities_test': int(np.isinf(test[audit_numeric].to_numpy(float)).sum())}
assert not checks['numeric_infinities_train'] and not checks['numeric_infinities_test']
display(styled_table(quality))
display(styled_table(pd.DataFrame({'check': checks.keys(), 'result': [str(x) for x in checks.values()]}).set_index('check')))
display(callout(f"The training dataset contains {len(train):,} rows, of which "
                f"{int(train[target].sum()):,} are labelled as fraud ({train[target].mean():.3%}). "
                "No threshold is needed to calculate ROC AUC."))

# %% cell 21
# KS is a distance between empirical CDFs, including for tied/discrete values.
# We compare distances, not significance p-values from huge samples.
shift_rows = []
for column in audit_numeric:
    shift_rows.append({'feature': column,
        'source to train': ks_2samp(original[column], train[column]).statistic,
        'train to test': ks_2samp(train[column], test[column]).statistic,
        'source median': original[column].median(),
        'train median': train[column].median(), 'test median': test[column].median()})
numeric_shifts = pd.DataFrame(shift_rows).set_index('feature')
category_shifts = []
for column in audit_categories:
    frequencies = {name: f[column].astype('string').fillna('<missing>').value_counts(normalize=True)
                   for name, f in frames.items()}
    support = frequencies['Original'].index.union(frequencies['S6E11 train'].index).union(frequencies['S6E11 test'].index)
    values = {name: freq.reindex(support, fill_value=0) for name, freq in frequencies.items()}
    category_shifts.append({'feature': column,
        'source to train': float((values['Original'] - values['S6E11 train']).abs().sum()/2),
        'train to test': float((values['S6E11 train'] - values['S6E11 test']).abs().sum()/2)})
category_shifts = pd.DataFrame(category_shifts).set_index('feature')
display(styled_table(numeric_shifts.sort_values('source to train', ascending=False),
                    {c: '{:.4f}' for c in numeric_shifts.columns}))
display(styled_table(category_shifts.sort_values('source to train', ascending=False),
                    {c: '{:.4f}' for c in category_shifts.columns}))

# %% cell 22
ordered = numeric_shifts.sort_values('source to train')
fig, ax = plt.subplots(figsize=(7.3, 6.4))
ax.barh([x.replace('_', ' ') for x in ordered.index], ordered['source to train'])
ax.set(xlabel='KS distance: original source to S6E11 train', title='The amount distribution moved most')
ax.grid(axis='x'); fig.tight_layout()
show_figure(fig, 'All 13 numeric predictors are shown. Amount has KS distance '
    f"{numeric_shifts.loc['amount_usd', 'source to train']:.3f}; the largest train-to-test distance is "
    f"{numeric_shifts['train to test'].max():.4f}. These are marginal comparisons, not a guarantee that CV will match the leaderboard.",
    'figures/numeric_shift.png')

# %% cell 25
def wilson(positives, count, z=1.959963984540054):
    p = positives / count
    centre = (p + z*z/(2*count)) / (1 + z*z/count)
    half = z*np.sqrt(p*(1-p)/count + z*z/(4*count*count)) / (1 + z*z/count)
    return centre-half, centre+half

strata_rows = []
for column in audit_categories:
    for level, subset in train.groupby(column, observed=True, dropna=False):
        count, positives = len(subset), int(subset[target].sum())
        low, high = wilson(positives, count)
        strata_rows.append(dict(feature=column, level=str(level), rows=count, fraud=positives,
            rate=positives/count, low=low, high=high))
strata = pd.DataFrame(strata_rows)
assert strata.groupby('feature').rows.sum().eq(len(train)).all()
assert strata.groupby('feature').fraud.sum().eq(int(train[target].sum())).all()
# All category/flag strata are saved; choose the displayed category in CFG.
strata.to_csv('category_fraud_rates.csv', index=False)
selected = CFG['category_view']
if selected not in audit_categories:
    raise ValueError('category_view must name a categorical or boolean predictor.')
selected_strata = strata[strata.feature.eq(selected)].copy()
display(styled_table(selected_strata.drop(columns='feature').set_index('level'),
    {'rows':'{:,.0f}', 'fraud':'{:,.0f}', 'rate':'{:.2%}', 'low':'{:.2%}', 'high':'{:.2%}'}))
flag_strata = strata[strata.feature.isin([c for c in audit_categories if pd.api.types.is_bool_dtype(train[c])])]
display(styled_table(flag_strata.set_index(['feature','level']),
    {'rows':'{:,.0f}', 'fraud':'{:,.0f}', 'rate':'{:.2%}', 'low':'{:.2%}', 'high':'{:.2%}'}))

# %% cell 26
true_flags = flag_strata[flag_strata.level.eq('True')].sort_values('rate')
short_names = {'is_foreign_transaction':'Foreign transaction', 'is_new_merchant':'New merchant',
    'used_vpn':'Used VPN', 'ip_country_mismatch':'IP country mismatch',
    'billing_shipping_mismatch':'Billing / shipping mismatch',
    'is_ai_generated_scam_attempt':'AI scam flag'}
fig, ax = plt.subplots(figsize=(7.3, 4.7))
positions = np.arange(len(true_flags))
ax.errorbar(true_flags.rate*100, positions,
    xerr=np.vstack([(true_flags.rate-true_flags.low)*100, (true_flags.high-true_flags.rate)*100]),
    fmt='o', capsize=4, color='#0072B2')
ax.set_yticks(positions, [short_names[c] for c in true_flags.feature])
ax.axvline(train[target].mean()*100, color='#D55E00', ls='--', label='All training rows')
ax.set(xlabel='Fraud share when flag is True (%)', title='Several flags mark higher-risk subsets')
ax.grid(axis='x'); ax.legend(loc='lower right'); fig.tight_layout()
show_figure(fig, 'Points show marginal fraud rates; bars are 95% Wilson intervals for each subset. '
    'Counts and both flag states are in the table. AI=True contains 350 fraud labels among 2,299 rows (15.22%). '
    'Correlated features and selection can explain these associations; the chart does not establish causality.',
    'figures/flag_rates.png')

# %% cell 28
# ID order is not documented event time. Check it before inventing a temporal split.
id_auc = roc_auc_score(train[target], train[id_col])
id_bins = pd.qcut(train[id_col], 20, labels=False, duplicates='drop')
id_summary = train.groupby(id_bins, observed=True)[target].agg(['count', 'sum', 'mean'])
print(f'ID as a score: ROC AUC {id_auc:.5f}; 20 ordered-bin fraud shares '
      f"{id_summary['mean'].min():.3%} to {id_summary['mean'].max():.3%}.")

# %% cell 31
if ready:
    encoder = None
    probability_order = None
    if task == "regression":
        y = train[target].to_numpy(dtype=float)
        if not np.isfinite(y).all() or (metric == "rmsle" and (y < 0).any()):
            raise ValueError("Invalid target for the configured regression metric.")
    else:
        encoder = LabelEncoder().fit(train[target])
        y = encoder.transform(train[target])
        if (task == "binary" and len(encoder.classes_) != 2) or (task == "multiclass" and len(encoder.classes_) < 3):
            raise ValueError("Task and observed class count disagree.")
        if task == "binary":
            if CFG["positive_class"] not in encoder.classes_:
                raise ValueError("Set positive_class explicitly, with the same type as the training labels.")
            y = (train[target].to_numpy() == CFG["positive_class"]).astype(int)
            probability_order = [next(c for c in encoder.classes_ if c != CFG["positive_class"]), CFG["positive_class"]]
        else:
            probability_order = encoder.classes_.tolist()
        if pd.Series(y).value_counts().min() < CFG["n_splits"]:
            raise ValueError("Too few examples of a class for n_splits.")
    splitter = (KFold(CFG["n_splits"], shuffle=True, random_state=CFG["seed"]) if task == "regression"
                else StratifiedKFold(CFG["n_splits"], shuffle=True, random_state=CFG["seed"]))
    folds = list(splitter.split(train, y))
    fold_ids = np.full(len(train), -1, dtype=np.int16)
    for fold, (_, validation) in enumerate(folds):
        fold_ids[validation] = fold
    data_fp = hashlib.sha256(pd.util.hash_pandas_object(train[[id_col, target]], index=False).values.tobytes()).hexdigest()
    split_fp = hashlib.sha256(data_fp.encode() + fold_ids.tobytes()).hexdigest()
    print("Split: IID", CFG["n_splits"], "folds, seed", CFG["seed"], "fingerprint", split_fp[:12])

    def score(actual, prediction):
        if metric == "auc": return float(roc_auc_score(actual, prediction[:, 1]))
        if metric == "logloss": return float(log_loss(actual, prediction, labels=np.arange(prediction.shape[1])))
        if metric == "accuracy": return float(accuracy_score(actual, prediction.argmax(axis=1)))
        if metric == "balanced_accuracy": return float(balanced_accuracy_score(actual, prediction.argmax(axis=1)))
        if metric == "map@3":
            top = np.argsort(-prediction, axis=1, kind="stable")[:, :3]
            return float(((top == actual[:, None]) / np.arange(1, top.shape[1] + 1)).sum(axis=1).mean())
        if metric == "mae": return float(mean_absolute_error(actual, prediction))
        if metric == "rmsle": return float(np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(np.maximum(prediction, 0)))))
        return float(np.sqrt(mean_squared_error(actual, prediction)))

# %% cell 33
if ready:
    shape = () if task == "regression" else (len(np.unique(y)),)
    oof = np.full((len(train),) + shape, np.nan)
    test_prediction = np.zeros((len(test),) + shape)
    fold_scores = []
    cat_indices = list(range(len(numeric), len(features)))

    def matrix(frame, category_encoder=None, fit=False):
        numbers = frame[numeric].to_numpy(dtype=np.float32)
        if not categorical: return pd.DataFrame(numbers, columns=numeric, index=frame.index), None
        categories = frame[categorical].astype("string").fillna("<missing>").astype(str)
        if fit:
            category_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            codes = category_encoder.fit_transform(categories)
        else:
            codes = category_encoder.transform(categories)
        result = pd.DataFrame(np.column_stack([numbers, codes]).astype(np.float32),
                              columns=numeric + categorical, index=frame.index)
        return result, category_encoder

    for fold, (fitting, validation) in enumerate(folds):
        X_fit, category_encoder = matrix(train.iloc[fitting], fit=True)
        X_valid, _ = matrix(train.iloc[validation], category_encoder)
        X_test, _ = matrix(test, category_encoder)
        parameters = dict(n_estimators=CFG["n_estimators"], num_leaves=CFG["num_leaves"],
                          learning_rate=CFG["learning_rate"], random_state=CFG["seed"],
                          n_jobs=CFG["n_jobs"], verbosity=-1, deterministic=True, force_col_wise=True)
        if task == "regression":
            model = LGBMRegressor(**parameters, objective="regression_l1" if metric == "mae" else "regression")
            fit_y = np.log1p(y[fitting]) if metric == "rmsle" else y[fitting]
        else:
            model = LGBMClassifier(**parameters)
            fit_y = y[fitting]
        model.fit(X_fit, fit_y, categorical_feature=cat_indices)
        if task == "regression":
            valid_pred, test_pred = model.predict(X_valid), model.predict(X_test)
            if metric == "rmsle":
                valid_pred, test_pred = np.maximum(np.expm1(valid_pred), 0), np.maximum(np.expm1(test_pred), 0)
        else:
            valid_pred, test_pred = model.predict_proba(X_valid), model.predict_proba(X_test)
        oof[validation] = valid_pred
        test_prediction += test_pred / len(folds)
        fold_scores.append(score(y[validation], valid_pred))
        print(f"Fold {fold}: {metric} = {fold_scores[-1]:.7f}")
    if not np.isfinite(oof).all() or not np.isfinite(test_prediction).all():
        raise ValueError("Predictions are incomplete or nonfinite.")
    pooled_score = score(y, oof)
    print(f"Pooled OOF {metric}: {pooled_score:.7f}")
    print(f"Fold mean: {np.mean(fold_scores):.7f}; fold SD: {np.std(fold_scores, ddof=1):.7f}")
    print("Fold SD describes variation between folds. It is not a significance threshold for a model change.")

# %% cell 35
"""Fixed-budget experiments on exactly the baseline's held-out folds."""
import time
research_started = time.perf_counter()
experiment_rows = [dict(recipe='Baseline', auc=pooled_score, delta=0.0)]
experiment_fold_deltas = {}
research_predictions = {}

def experiment_matrices(fitting_frame, other_frames, columns):
    cats = [c for c in columns if not pd.api.types.is_numeric_dtype(fitting_frame[c])]
    nums = [c for c in columns if c not in cats]
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if cats:
        enc.fit(fitting_frame[cats].astype('string').fillna('<missing>').astype(str))
    result = []
    for frame in [fitting_frame] + other_frames:
        values = frame[nums].to_numpy(dtype=np.float32)
        if cats:
            codes = enc.transform(frame[cats].astype('string').fillna('<missing>').astype(str))
            values = np.column_stack([values, codes]).astype(np.float32)
        result.append(pd.DataFrame(values, columns=nums+cats, index=frame.index))
    return result, list(range(len(nums), len(columns)))

def frequency_matrices(fitting_frame, other_frames, columns):
    # Only the fitting rows determine each map. No labels enter the encoding.
    cats = [c for c in columns if not pd.api.types.is_numeric_dtype(fitting_frame[c])]
    nums = [c for c in columns if c not in cats]
    maps = {c: fitting_frame[c].astype('string').fillna('<missing>')
              .value_counts(normalize=True) for c in cats}
    result = []
    for frame in [fitting_frame] + other_frames:
        values = frame[nums].to_numpy(dtype=np.float32)
        codes = np.column_stack([
            frame[c].astype('string').fillna('<missing>').map(maps[c])
                 .fillna(0.0).to_numpy(dtype=np.float32) for c in cats]) if cats else np.empty((len(frame), 0))
        result.append(pd.DataFrame(np.column_stack([values, codes]).astype(np.float32),
                                   columns=nums+cats, index=frame.index))
    # Frequencies are ordinary numeric inputs, with no native categorical columns.
    return result, []

def evaluate_variant(name, fitting_frame, test_frame, columns, representation='native'):
    variant_oof = np.full(len(train), np.nan)
    variant_test = np.zeros(len(test))
    deltas = []
    for fold, (fitting, validation) in enumerate(folds):
        transform = frequency_matrices if representation == 'frequency' else experiment_matrices
        (a, b, c), cat_idx = transform(
            fitting_frame.iloc[fitting], [fitting_frame.iloc[validation], test_frame], columns)
        fitted = LGBMClassifier(**parameters).fit(a, y[fitting], categorical_feature=cat_idx)
        variant_oof[validation] = fitted.predict_proba(b)[:, 1]
        variant_test += fitted.predict_proba(c)[:, 1] / len(folds)
        deltas.append(roc_auc_score(y[validation], variant_oof[validation])
                      - roc_auc_score(y[validation], oof[validation, 1]))
        print(f'{name}, fold {fold}: delta {deltas[-1]:+.6f}')
    assert np.isfinite(variant_oof).all() and np.isfinite(variant_test).all()
    auc = roc_auc_score(y, variant_oof)
    experiment_rows.append(dict(recipe=name, auc=auc, delta=auc-pooled_score))
    experiment_fold_deltas[name] = deltas
    research_predictions[name] = (variant_oof, variant_test)

if CFG['run_research']:
    # No competition labels enter this teacher. Exact feature-row matches are excluded.
    source_hash = pd.util.hash_pandas_object(original[schema_features], index=False)
    competition_hash = pd.util.hash_pandas_object(
        pd.concat([train[schema_features], test[schema_features]], ignore_index=True), index=False)
    source_overlap = source_hash.isin(competition_hash)
    source_fit = original.loc[~source_overlap]
    print('Source rows excluded by full-feature hash match:', int(source_overlap.sum()))
    (a, b, c), cat_idx = experiment_matrices(source_fit, [train, test], features)
    teacher_parameters = {**parameters, 'n_estimators': 100, 'num_leaves': 15}
    teacher = LGBMClassifier(**teacher_parameters).fit(
        a, source_fit[target].astype(int), categorical_feature=cat_idx)
    teacher_train, teacher_test = teacher.predict_proba(b)[:, 1], teacher.predict_proba(c)[:, 1]
    teacher_auc = roc_auc_score(y, teacher_train)
    print(f'Source-only teacher AUC: {teacher_auc:.7f}')
    evaluate_variant('Add source teacher',
        train.assign(source_teacher_probability=teacher_train),
        test.assign(source_teacher_probability=teacher_test),
        features + ['source_teacher_probability'])
    flag = 'is_ai_generated_scam_attempt'
    if flag in features:
        evaluate_variant('Drop AI flag', train, test, [c for c in features if c != flag])
    else:
        print('AI flag already excluded in CFG; its ablation is skipped.')
    print('Frequency-encoded columns:', categorical)
    evaluate_variant('Frequency encoding', train, test, features, representation='frequency')
else:
    print('Research disabled in CFG. The baseline still runs and exports predictions.')
experiment_results = pd.DataFrame(experiment_rows).set_index('recipe')
display(styled_table(experiment_results, {'auc':'{:.7f}', 'delta':'{:+.7f}'}))
if experiment_fold_deltas:
    display(styled_table(pd.DataFrame(experiment_fold_deltas).rename_axis('fold'),
                         {name:'{:+.6f}' for name in experiment_fold_deltas}))
print(f'Research cells: {time.perf_counter()-research_started:.1f} seconds on this machine.')

# %% cell 36
"""Plot measured comparisons; this cell does not fit or select a model."""
if experiment_fold_deltas:
    names = list(experiment_fold_deltas)
    labels = {'Add source teacher': 'Source teacher', 'Drop AI flag': 'Drop AI flag',
              'Frequency encoding': 'Frequency encoding'}
    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    for row, name in enumerate(names):
        values = np.asarray(experiment_fold_deltas[name])
        ax.scatter(values, row + np.linspace(-.11, .11, len(values)),
                   s=37, color='#0072B2', alpha=.8,
                   label='Individual folds' if row == 0 else None)
        ax.scatter(experiment_results.loc[name, 'delta'], row, marker='D',
                   s=65, color='#14212B', zorder=4,
                   label='Pooled OOF' if row == 0 else None)
    ax.axvline(0, color='#52616E', linewidth=1, linestyle='--')
    ax.set_yticks(range(len(names)), [labels.get(name, name) for name in names])
    ax.set_ylim(-.5, len(names)-.5)
    ax.invert_yaxis()
    ax.set(title='Change from the native-category baseline', xlabel='AUC difference')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:+.3f}'))
    ax.grid(axis='x')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.30), ncol=2, fontsize=14)
    fig.subplots_adjust(left=.32, right=.97, top=.82, bottom=.35)
    show_figure(fig,
        'Small circles show each fold difference; black diamonds show pooled OOF differences. '
        'Positive values favour the variant. Pooled AUC compares all held-out rows together, '
        'so its difference is not the mean of the five fold differences. '
        'The folds share training rows and are not independent replications.',
        'experiment_comparisons.png')

# %% cell 38
"""Repeat the complete frequency pipeline after shuffling labels."""
frequency_control = None
if CFG['run_research']:
    shuffled_y = np.random.RandomState(0).permutation(y)
    # This control re-stratifies after shuffling, like the earlier LightGBM control.
    # The true-label folds used for the three model comparisons stay unchanged.
    control_folds = list(StratifiedKFold(CFG['n_splits'], shuffle=True,
                                       random_state=CFG['seed']).split(train, shuffled_y))
    control_oof = np.full(len(train), np.nan)
    control_fold_ids = np.full(len(train), -1, dtype=np.int16)
    control_scores = []
    for k, (fitting, validation) in enumerate(control_folds):
        (a, b), cats = frequency_matrices(train.iloc[fitting],
                                         [train.iloc[validation]], features)
        fitted = LGBMClassifier(**parameters).fit(a, shuffled_y[fitting],
                                                categorical_feature=cats)
        control_oof[validation] = fitted.predict_proba(b)[:, 1]
        control_fold_ids[validation] = k
        control_scores.append(roc_auc_score(shuffled_y[validation], control_oof[validation]))
        print(f'Frequency control, fold {k}: AUC {control_scores[-1]:.7f}')
    assert np.isfinite(control_oof).all() and (control_fold_ids >= 0).all()
    frequency_control = {'auc': float(roc_auc_score(shuffled_y, control_oof)),
                         'fold_auc': [float(v) for v in control_scores],
                         'permutation_seed': 0, 're_stratified_after_shuffle': True}
    print(f"The shuffled-label frequency pipeline has pooled AUC {frequency_control['auc']:.7f}.")
    print('This single control cannot certify absence of leakage or event-time feature availability.')
else:
    print('The frequency control is skipped because run_research is False.')

# %% cell 41
if ready:
    submission = sample.copy()
    prediction_columns = [c for c in sample if c != id_col]
    order = pd.Index(test[id_col]).get_indexer(sample[id_col])
    aligned = test_prediction[order]
    if task == "regression":
        if len(prediction_columns) != 1: raise ValueError("Expected one regression prediction column.")
        submission[prediction_columns[0]] = aligned
    elif task == "binary" and metric in {"auc", "logloss"} and len(prediction_columns) == 1:
        submission[prediction_columns[0]] = aligned[:, 1]
    elif metric in {"accuracy", "balanced_accuracy", "map@3"} and len(prediction_columns) == 1:
        if task == "binary":
            negative = next(c for c in encoder.classes_ if c != CFG["positive_class"])
            labels = np.array([negative, CFG["positive_class"]])
        else: labels = encoder.classes_
        if metric == "map@3":
            indices = np.argsort(-aligned, axis=1, kind="stable")[:, :3]
            submission[prediction_columns[0]] = [" ".join(map(str, labels[row])) for row in indices]
        else: submission[prediction_columns[0]] = labels[aligned.argmax(axis=1)]
    else:
        mapping = CFG["class_columns"]
        if task != "multiclass" or not mapping or set(mapping) != set(prediction_columns):
            raise ValueError("Set class_columns as {submission_column: class_label} for class probabilities.")
        if set(mapping.values()) != set(encoder.classes_):
            raise ValueError("class_columns must cover each class exactly once.")
        for column in prediction_columns:
            submission[column] = aligned[:, int(encoder.transform([mapping[column]])[0])]
    if submission[prediction_columns].isna().any().any(): raise ValueError("Submission contains missing predictions.")
    output = Path(os.environ.get("S6E11_OUTPUT", "/kaggle/working" if Path("/kaggle/working").is_dir() else "artifacts/baseline"))
    output.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output / "submission.csv", index=False)
    pd.DataFrame({id_col: train[id_col], "fold": fold_ids}).to_csv(output / "folds.csv", index=False)
    np.save(output / "oof.npy", oof)
    np.save(output / "test.npy", test_prediction)
    report = {"config": CFG, "data_fp": data_fp, "split_fp": split_fp, "pooled_oof": pooled_score,
              "fold_scores": fold_scores, "features": features, "checks": checks,
              "classes": None if encoder is None else encoder.classes_.tolist(),
              "probability_order": None if probability_order is None else [str(c) for c in probability_order],
              "libraries": {name: importlib.metadata.version(name)
                            for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "matplotlib")}}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved artifacts:", output.resolve())
    display(submission.head())

# %% cell 42
import shutil
shutil.copyfile(output/'submission.csv',output/'reference_submission.csv')
s10_reference_metrics=json.loads((output/'metrics.json').read_text())
s10_reference_metrics['artifact_scope']='Reference model only; submission_manifest.json identifies the final primary CSV and its model report.'
(output/'metrics.json').write_text(json.dumps(s10_reference_metrics,indent=2))
(output/'reference_metrics.json').write_text(json.dumps(s10_reference_metrics,indent=2))
print('Reference export preserved as reference_submission.csv. Section 17 writes the primary current-model file.')

# %% cell 43
for name, (variant_oof, variant_test) in research_predictions.items():
    stem = name.lower().replace(' ', '_')
    np.save(output / (stem + '_oof.npy'), variant_oof)
    np.save(output / (stem + '_test.npy'), variant_test)
if frequency_control is not None:
    np.save(output / 'frequency_control_oof.npy', control_oof)
    np.save(output / 'frequency_control_labels.npy', shuffled_y)
    np.save(output / 'frequency_control_fold_ids.npy', control_fold_ids)
    (output / 'frequency_control.json').write_text(json.dumps(frequency_control, indent=2))
experiment_results.to_csv(output / 'experiment_results.csv')
pd.DataFrame(experiment_fold_deltas).to_csv(output / 'experiment_fold_deltas.csv', index_label='fold')
print(f'Elapsed time through the reference export: {time.perf_counter()-notebook_started:.1f} seconds.')

# %% cell 46
# A fixed score under random assignment of n1 positive labels has the Mann-Whitney
# null SD below (untied ranks). This is a reference, not a refitted-OOF null law.
n1 = int(y.sum()); n0 = len(y)-n1
fixed_score_null_sd = np.sqrt((n0+n1+1)/(12*n0*n1))
print(f'Fixed-score, untied-rank reference SD: {fixed_score_null_sd:.5f}')
print(f'A +/-0.003 band is only {0.003/fixed_score_null_sd:.2f} reference SDs wide.')

# %% cell 49
recorded_controls = {'recorded_date': '1 November 2026',
 'scope': 'Two separately measured protocols; default display does not refit them.',
 'seeds': [11, 23, 47, 101, 211],
 'rule': 'abs(AUC-.5)<.003, unchanged',
 'pipelines': {'frequency': {'label': 'LightGBM with fitting-fold frequencies',
                             'fold_protocol': 'Re-stratify each permuted label vector, 5 '
                                              'folds, split seed42',
                             'device': 'CPU',
                             'model': {'n_estimators': 300,
                                       'num_leaves': 31,
                                       'learning_rate': 0.05,
                                       'random_state': 42,
                                       'n_jobs': 4,
                                       'verbosity': -1,
                                       'deterministic': True,
                                       'force_col_wise': True},
                             'libraries': {'lightgbm': '4.6.0',
                                           'numpy': '2.2.6',
                                           'pandas': '2.2.3',
                                           'scikit-learn': '1.6.1'},
                             'train_sha256': '87eaa10b8d17a7e4f657b8da1c66242431338bdd8b6ad45861521ac0c0e9fbd5',
                             'private_report_sha256': '0445fcd3b678a8357f94e9b05e7154d124d83241cfcbbb9d86335f13977098d9',
                             'arms': [{'seed': 11,
                                       'control': 'shuffle',
                                       'auc': 0.49925097751324077,
                                       'fold_auc': [0.501899894322263,
                                                    0.5044101411303079,
                                                    0.4964144784950273,
                                                    0.49194702229016374,
                                                    0.502051737360168]},
                                      {'seed': 23,
                                       'control': 'shuffle',
                                       'auc': 0.4999769529649386,
                                       'fold_auc': [0.49882136987150455,
                                                    0.502196044202171,
                                                    0.5033258263615114,
                                                    0.5022828200204867,
                                                    0.4941099299525646]},
                                      {'seed': 47,
                                       'control': 'shuffle',
                                       'auc': 0.4903986519044923,
                                       'fold_auc': [0.49056051600415523,
                                                    0.4856697057348496,
                                                    0.49274912261576775,
                                                    0.49152356313258255,
                                                    0.4914487453032307]},
                                      {'seed': 101,
                                       'control': 'shuffle',
                                       'auc': 0.5025771868597659,
                                       'fold_auc': [0.49919775812062767,
                                                    0.5058995997738618,
                                                    0.5061098146015494,
                                                    0.4927213552860613,
                                                    0.5096328957356702]},
                                      {'seed': 211,
                                       'control': 'shuffle',
                                       'auc': 0.5002233806571335,
                                       'fold_auc': [0.5000682282190876,
                                                    0.5066768628421493,
                                                    0.4878527557029431,
                                                    0.500113528653181,
                                                    0.5064020434915212]},
                                      {'seed': 11,
                                       'control': 'target copy leak',
                                       'auc': 1.0,
                                       'fold_auc': [1.0000000000000002,
                                                    1.0,
                                                    1.0,
                                                    0.9999999999999999,
                                                    1.0]}]},
               'xgboost': {'label': 'Historical XGBoost with cross-fitted target '
                                    'encoding',
                           'fold_protocol': 'Original true-label stratified folds held '
                                            'fixed under all permutations',
                           'device': 'cuda',
                           'model': {'n_estimators': 1200,
                                     'learning_rate': 0.02,
                                     'max_depth': 6,
                                     'min_child_weight': 5,
                                     'subsample': 0.8,
                                     'colsample_bytree': 0.4,
                                     'reg_lambda': 2.0,
                                     'reg_alpha': 0.1,
                                     'max_bin': 1024,
                                     'tree_method': 'hist',
                                     'device': 'cuda',
                                     'eval_metric': 'auc',
                                     'random_state': 42,
                                     'n_jobs': 4},
                           'libraries': {'xgboost': '3.4.1',
                                         'scikit-learn': '1.6.1',
                                         'numpy': '2.1.3',
                                         'pandas': '2.3.3',
                                         'scipy': '1.16.3'},
                           'train_sha256': '87eaa10b8d17a7e4f657b8da1c66242431338bdd8b6ad45861521ac0c0e9fbd5',
                           'private_report_sha256': 'b2f48efe7417be7cb0464214e91667cd3aae810988289a883712c90234c83c4f',
                           'arms': [{'seed': 11,
                                     'control': 'shuffle',
                                     'auc': 0.5037117612418591,
                                     'fold_auc': [0.5035042044399397,
                                                  0.493227635673,
                                                  0.5070754373127241,
                                                  0.5013884384815226,
                                                  0.5137934035202791]},
                                    {'seed': 23,
                                     'control': 'shuffle',
                                     'auc': 0.497128319394553,
                                     'fold_auc': [0.49459901243225535,
                                                  0.49919826637860687,
                                                  0.4959514887384038,
                                                  0.5013210764845434,
                                                  0.4942300724195321]},
                                    {'seed': 47,
                                     'control': 'shuffle',
                                     'auc': 0.5015362399233932,
                                     'fold_auc': [0.49948773636567756,
                                                  0.49596461972823336,
                                                  0.49795944227273914,
                                                  0.507017367748154,
                                                  0.5082105925711986]},
                                    {'seed': 101,
                                     'control': 'shuffle',
                                     'auc': 0.4980984725010778,
                                     'fold_auc': [0.4918950479329962,
                                                  0.49862399546903435,
                                                  0.501241118926723,
                                                  0.50017661782656,
                                                  0.49932298401953795]},
                                    {'seed': 211,
                                     'control': 'shuffle',
                                     'auc': 0.49955263372871894,
                                     'fold_auc': [0.5084871062239789,
                                                  0.4978568428342482,
                                                  0.49929143904082895,
                                                  0.4977667052346681,
                                                  0.4951262233909697]},
                                    {'seed': 11,
                                     'control': 'validation TE leak',
                                     'auc': 0.46792899579725156,
                                     'fold_auc': [0.46599409814133413,
                                                  0.4648408749621926,
                                                  0.4769919818727415,
                                                  0.461641852915222,
                                                  0.4684302383620612]}]}}}
reproduction_sources = {'run_frequency_controls.py': '"""Optional S6E11 frequency-control reproduction. No submission is uploaded.\n\nThis file contains the same fold-fitted frequency_matrices function as section 8.\nCall run_repeated_frequency(data_directory, output_directory) to run six arms.\nThe default notebook displays a recorded snapshot and leaves this rerun disabled.\n"""\nfrom pathlib import Path\nfrom datetime import datetime, timezone\nimport hashlib, json, time, importlib.metadata\nimport numpy as np\nimport pandas as pd\nfrom lightgbm import LGBMClassifier\nfrom sklearn.metrics import roc_auc_score\nfrom sklearn.model_selection import StratifiedKFold\n\n# The builder inserts the exact frequency_matrices definition at this marker.\ndef frequency_matrices(fitting_frame, other_frames, columns):\n    # Only the fitting rows determine each map. No labels enter the encoding.\n    cats = [c for c in columns if not pd.api.types.is_numeric_dtype(fitting_frame[c])]\n    nums = [c for c in columns if c not in cats]\n    maps = {c: fitting_frame[c].astype(\'string\').fillna(\'<missing>\')\n              .value_counts(normalize=True) for c in cats}\n    result = []\n    for frame in [fitting_frame] + other_frames:\n        values = frame[nums].to_numpy(dtype=np.float32)\n        codes = np.column_stack([\n            frame[c].astype(\'string\').fillna(\'<missing>\').map(maps[c])\n                 .fillna(0.0).to_numpy(dtype=np.float32) for c in cats]) if cats else np.empty((len(frame), 0))\n        result.append(pd.DataFrame(np.column_stack([values, codes]).astype(np.float32),\n                                   columns=nums+cats, index=frame.index))\n    # Frequencies are ordinary numeric inputs, with no native categorical columns.\n    return result, []\n\ndef run_repeated_frequency(data, output_directory):\n    output_directory = Path(output_directory)\n    output_directory.mkdir(parents=True, exist_ok=True)\n    parameters = dict(n_estimators=300, num_leaves=31, learning_rate=.05,\n                      random_state=42, n_jobs=4, verbosity=-1,\n                      deterministic=True, force_col_wise=True)\n    seeds = [11, 23, 47, 101, 211]\n    protocol = {\'created_utc\': datetime.now(timezone.utc).isoformat(), \'seeds\': seeds,\n        \'model\': parameters, \'folds\': \'Re-stratify each shuffled label vector, five folds, seed42\',\n        \'encoding\': \'Fitting-row-only text frequencies; unknown0; boolean inputs unchanged\',\n        \'positive_control\': \'Seed11: add a numeric input equal to the shuffled target, including held-out rows\',\n        \'gate\': \'abs(AUC-.5)<.003 unchanged; no admission or calibration from five repeats\',\n        \'train_file_sha256\': hashlib.sha256((Path(data)/\'train.csv\').read_bytes()).hexdigest(),\n        \'libraries\': {p: importlib.metadata.version(p) for p in [\'lightgbm\', \'numpy\', \'pandas\', \'scikit-learn\']}}\n    (output_directory/\'protocol.json\').write_text(json.dumps(protocol, indent=2))\n    train = pd.read_csv(Path(data)/\'train.csv\')\n    true_y = train.is_fraud.to_numpy()\n    assert len(train) == 699960 and true_y.sum() == 11689\n    features = [c for c in train if c not in {\'id\', \'is_fraud\'}]\n    report = {\'status\': \'running\', \'protocol\': protocol, \'arms\': []}\n    for seed, leak in [(s, False) for s in seeds] + [(11, True)]:\n        started = time.perf_counter()\n        name = (\'target_copy_\' if leak else \'null_\') + str(seed)\n        frame = train.copy() if leak else train\n        labels = np.random.RandomState(seed).permutation(true_y)\n        columns = features\n        if leak:\n            frame[\'deliberately_leaked_shuffled_target\'] = labels\n            columns = features + [\'deliberately_leaked_shuffled_target\']\n        oof = np.full(len(frame), np.nan)\n        fold_ids = np.full(len(frame), -1, np.int16)\n        fold_scores = []\n        for k, (fit, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(frame, labels)):\n            (a, b), cats = frequency_matrices(frame.iloc[fit], [frame.iloc[valid]], columns)\n            model = LGBMClassifier(**parameters).fit(a, labels[fit], categorical_feature=cats)\n            oof[valid] = model.predict_proba(b)[:, 1]\n            fold_ids[valid] = k\n            fold_scores.append(float(roc_auc_score(labels[valid], oof[valid])))\n            print(f\'{name}, fold {k}: AUC {fold_scores[-1]:.7f}\', flush=True)\n        assert np.isfinite(oof).all() and (fold_ids >= 0).all()\n        np.savez_compressed(output_directory/(name+\'.npz\'), oof=oof, labels=labels, fold_ids=fold_ids)\n        auc = float(roc_auc_score(labels, oof))\n        report[\'arms\'].append({\'name\': name, \'seed\': seed, \'deliberate_leak\': leak, \'auc\': auc,\n            \'fold_auc\': fold_scores, \'old_gate_pass\': bool(abs(auc-.5) < .003),\n            \'seconds\': time.perf_counter()-started,\n            \'fold_ids_sha256\': hashlib.sha256(fold_ids.tobytes()).hexdigest(),\n            \'labels_sha256\': hashlib.sha256(labels.tobytes()).hexdigest()})\n        (output_directory/\'report.json\').write_text(json.dumps(report, indent=2))\n    report[\'status\'] = \'complete\'\n    (output_directory/\'report.json\').write_text(json.dumps(report, indent=2))\n    return report\n', 'run_h4_controls.py': '# Optional H4 reproduction: historical transductive pipeline diagnostics.\n# Global train+test feature-only statistics differ from the fold-only H3 recipe.\n"""Optional H4 controls for the historical transductive XGBoost pipeline.\nThis source defines run_controls and leaves expensive refits disabled by default.\nStandalone use after saving as run_h4_controls.py:\n    import run_h4_controls\n    run_h4_controls.run_controls(data=\'DATA_DIR\', out=\'OUTPUT_DIR\', device=\'cuda\')\nNotebook use: execute this source in an isolated namespace, then explicitly call\nthat namespace\'s run_controls function. Do not execute it in the baseline\'s\nnamespace: the historical adapter deliberately owns y, FOLDS and model settings.\nChanging RUN_REPEATED_NULL to True also runs all 30 costly fits in a dedicated\nnotebook cell. No command-line argument parser is provided by this adapter.\n"""\nimport os, glob, json, time, hashlib, platform, importlib.metadata, gc\nfrom pathlib import Path\nfrom datetime import datetime, timezone\nimport numpy as np, pandas as pd\nimport xgboost as xgb\nfrom sklearn.model_selection import StratifiedKFold, KFold\nfrom sklearn.preprocessing import TargetEncoder\nRUN_REPEATED_NULL = False\nTASK = \'binary\'\n_TE_TYPE = \'binary\'\nMETRICS = {\'auc\': (lambda y, p: __import__(\'sklearn.metrics\', fromlist=[\'roc_auc_score\']).roc_auc_score(y, p), True)}\n\ndef build_y(train, comp):\n    """Construct the target according to the binary task contract."""\n    s = train[comp[\'target\']]\n    if comp[\'task\'] == \'binary\':\n        pos = comp.get(\'positive\')\n        return (s == pos).astype(int).to_numpy() if pos is not None else s.astype(int).to_numpy()\n    if comp[\'task\'] == \'multiclass\':\n        idx = {c: i for i, c in enumerate(comp[\'classes\'])}\n        yy = s.map(lambda v: idx.get(v, idx.get(str(v))))\n        assert yy.notna().all(), \'Training data contain a class absent from the task contract.\'\n        return yy.astype(int).to_numpy()\n    return s.astype(float).to_numpy()\n\ndef make_folds(y, comp, n_splits=None, seed=None):\n    """Construct frozen folds: stratified for classification, KFold for regression."""\n    k = n_splits or comp.get(\'n_splits\', 5)\n    sd = comp.get(\'seed\', 42) if seed is None else seed\n    if comp[\'task\'] == \'regression\':\n        cv = KFold(k, shuffle=True, random_state=sd)\n    else:\n        cv = StratifiedKFold(k, shuffle=True, random_state=sd)\n    return list(cv.split(np.zeros(len(y)), y))\n\ndef fingerprint(folds, n):\n    ids = np.empty(n, np.int8)\n    for i, (_, va) in enumerate(folds):\n        ids[va] = i\n    return hashlib.md5(ids.tobytes()).hexdigest()[:12]\n\ndef _auc(y, p):\n    from sklearn.metrics import roc_auc_score\n    return roc_auc_score(y, p)\n\ndef score(y, p, comp):\n    return float(METRICS[comp[\'metric\']][0](y, p))\n\ndef pred_shape(comp, n):\n    return (n, len(comp[\'classes\'])) if comp[\'task\'] == \'multiclass\' else (n,)\nPROTOCOL = json.loads(\'{"experiment": "H4 repeated full-pipeline shuffled XGBoost controls", "seeds": [11, 23, 47, 101, 211], "positive_control": {"permutation_seed": 11, "mechanism": "historical encode_fold(leak_yB=y[outer_valid]): validation target encoder fit on A union B; training encodings remain inner-cross-fitted", "expected_detection": "absolute pooled AUC deviation >= .003; failure would limit sensitivity claims, not trigger another control tuned after viewing"}, "frozen_folds": "StratifiedKFold(5,shuffle=True,random_state=42) constructed once from original true labels before any permutation", "expected_factory_split_fp": "c7358c075703", "permutation": "np.random.RandomState(seed).permutation(n); permute both train[TARGET] and y identically, never re-stratify afterwards", "model_random_state": 42, "params": {"n_estimators": 1200, "learning_rate": 0.02, "max_depth": 6, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.4, "reg_lambda": 2.0, "reg_alpha": 0.1, "max_bin": 1024, "tree_method": "hist", "device": "cuda", "eval_metric": "auc", "random_state": 42, "n_jobs": 4}, "feature_recipe": "verbatim historical build_matrix and encode_fold; global train+test feature-only frequency/ladder construction; target dropped before build; inner 5-fold TargetEncoder for fitting, outer fitting-label-only encoder for validation in honest arms", "early_stopping": false, "n_jobs": 4, "sequential_arms": true, "gate": "retain abs(AUC-.5)<.003 unchanged; no candidate admission or submission", "analysis": "Report each pooled/fold AUC, signed deviations, descriptive mean/SD/range, exceedance count and positive control; five repeats do not calibrate a 5% tail or certify no leakage. Fixed historical tree budget; no seed or cutoff selection after viewing.", "input_file_sha256": {"data/train.csv": "87eaa10b8d17a7e4f657b8da1c66242431338bdd8b6ad45861521ac0c0e9fbd5", "data/test.csv": "10f516959020357b565abaaa4fd25baae97fc2f536d0805695bb7cc15c1660f0"}, "source_sha256": {"core.py": "c914308f1832d87620d8c23b9c5bd3ea53a53b04938e9a4103ffecc4e8aae349", "header.py": "987fd2a4a0f0c502d3540ea1d784844115b057d33440370d28f01ae26779f8f4", "lib_gbdt.py": "30a020012df59ddecf5173e46c065117a6e922acb33d9a64e7d07a2e3cb5ba33", "base_xgb_1200.py": "d838e2c7da96e769823a5784c4ab45dab0228e335af4659c0351c018314fde3a", "base_xgb_1200_shuf.py": "85383cb2dd0c20b41646a16670ca71cf20893053d60f52922d3e7f06a9902543", "comp.json": "b514c06dd6923ab8dfa65dcf169cf25278a09c0f5171c19d76e81219349b94d4", "meta.json": "7ea40c3d50ce0d5accc5b9d577078b41b5e25279f6338c095ad83003f1236e23"}, "provenance": "Frozen historical factory implementation; diagnostic adapter for the historical transductive pipeline, not an endorsed model recipe. All measurements are recreated from attached competition CSVs; no workshop arrays are loaded."}\')\nCOMP = json.loads(\'{"competition": "playground-series-s6e11", "task": "binary", "target": "is_fraud", "id_col": "id", "positive": null, "metric": "auc", "n_splits": 5, "seed": 42, "split_fp": "c7358c075703", "n_train": 699960, "n_test": 299984}\')\nTARGET = COMP[\'target\']\n\ndef build_matrix(train, test, extra=None, drop=(), flags=None, steps=(0.1, 1, 10, 100, 1000, 10000), lo=8, hi=20000):\n    """Build feature-only numeric ladder keys and transductive train+test frequencies."""\n    comb = pd.concat([train.assign(_tr=1), test.assign(_tr=0)], ignore_index=True)\n    comb.drop(columns=[TARGET], inplace=True, errors=\'ignore\')\n    added = extra(comb) if extra else []\n    comb.drop(columns=list(drop), inplace=True, errors=\'ignore\')\n    idc = COMP.get(\'id_col\', \'id\')\n    CAT = comb.select_dtypes(include=[\'object\', \'string\', \'category\']).columns.tolist()\n    NUM = [c for c in comb.columns if c not in CAT + [idc, \'_tr\'] + added]\n    LADDER = []\n    for c in NUM:\n        v = comb[c]\n        span = float(v.max()) - float(v.min())\n        for st in steps:\n            k = int(np.floor(v / st).nunique())\n            if lo <= k <= hi and st <= max(span, 1):\n                comb[f\'L_{c}_{st}\'] = np.floor(v / st).astype(str)\n                LADDER.append(f\'L_{c}_{st}\')\n    KEYS = CAT + LADDER\n    for c in KEYS:\n        comb[f\'{c}_fe\'] = comb[c].map(comb[c].value_counts(normalize=True).to_dict()).astype(float)\n    if flags:\n        flags(comb)\n    tr = comb[comb._tr == 1].drop(columns=[\'_tr\']).reset_index(drop=True)\n    te = comb[comb._tr == 0].drop(columns=[\'_tr\']).reset_index(drop=True)\n    const = [c for c in tr.columns if tr[c].nunique() == 1]\n    tr.drop(columns=const, inplace=True)\n    te.drop(columns=const, inplace=True)\n    FEATS = [c for c in te.columns if c != idc]\n    TEC = [c for c in KEYS if c in FEATS]\n    print(f\'Features: {len(FEATS)}, target-encoding keys: {len(TEC)}, added: {added}\', flush=True)\n    return (tr, te, FEATS, TEC)\n\ndef _te_put(X, col, tag, E, i, K):\n    if K == 1:\n        X[f\'{col}_TE_{tag}\'] = E[:, i].astype(\'float32\')\n    else:\n        for j in range(K):\n            X[f\'{col}_TE_{tag}_{j}\'] = E[:, i * K + j].astype(\'float32\')\n\ndef encode_fold(A, B, C, TEC, yA, leak_yB=None):\n    """Cross-fit training encodings; transform held-out rows using fitting labels only, unless explicitly injecting the leaky control."""\n    K = len(COMP[\'classes\']) if TASK == \'multiclass\' else 1\n    for tag, sm in ((\'auto\', \'auto\'), (\'10\', 10.0)):\n        e = TargetEncoder(target_type=_TE_TYPE, shuffle=True, cv=5, smooth=sm, random_state=42)\n        ea = e.fit_transform(A[TEC], yA)\n        ec = e.transform(C[TEC])\n        if len(B) == 0:\n            eb = np.zeros((0, ea.shape[1]))\n        elif leak_yB is None:\n            eb = e.transform(B[TEC])\n        else:\n            eL = TargetEncoder(target_type=_TE_TYPE, smooth=sm).fit(pd.concat([A[TEC], B[TEC]]), np.r_[yA, leak_yB])\n            eb = eL.transform(B[TEC])\n        for i, col in enumerate(TEC):\n            for X_, E_ in ((A, ea), (B, eb), (C, ec)):\n                _te_put(X_, col, tag, E_, i, K)\n    for X_ in (A, B, C):\n        X_.drop(columns=TEC, inplace=True)\n\ndef _link(z):\n    """Convert a raw model margin to a prediction in the metric contract."""\n    if TASK == \'binary\':\n        return 1 / (1 + np.exp(-z))\n    if TASK == \'multiclass\':\n        z = z - z.max(1, keepdims=True)\n        e = np.exp(z)\n        return e / e.sum(1, keepdims=True)\n    return z\n\ndef _xgb_model(P):\n    return xgb.XGBRegressor(**P) if TASK == \'regression\' else xgb.XGBClassifier(**P)\n\ndef _xgb_raw(m, X, margin):\n    kw = dict(base_margin=margin) if margin is not None else {}\n    z = m.predict(X, output_margin=True, **kw)\n    if TASK == \'multiclass\':\n        z = np.asarray(z).reshape(len(X), -1)\n    return z\n\ndef run_xgb(tr, te, FEATS, TEC, margin=None, P=None, note=\'\', fixed_iter=None, leak=False):\n    if globals().get(\'REFIT_MODE\'):\n        return _refit(tr, te, FEATS, TEC, \'xgb\', margin, P)\n    P = dict(XGB_P, **P or {})\n    if fixed_iter:\n        P.update(n_estimators=int(fixed_iter))\n        P.pop(\'early_stopping_rounds\', None)\n    oof = np.zeros(pred_shape(COMP, len(tr)))\n    tp = np.zeros(pred_shape(COMP, len(te)))\n    iters = []\n    for k, (i1, i2) in enumerate(FOLDS, 1):\n        A, B, C = (tr.iloc[i1][FEATS].copy(), tr.iloc[i2][FEATS].copy(), te[FEATS].copy())\n        encode_fold(A, B, C, TEC, y[i1], leak_yB=y[i2] if leak else None)\n        m = _xgb_model(P)\n        ma, mb, mc = margin(i1, i2) if margin is not None else (None, None, None)\n        kw = dict(base_margin=ma) if margin is not None else {}\n        if not fixed_iter:\n            kw.update(eval_set=[(B, y[i2])])\n            if margin is not None:\n                kw.update(base_margin_eval_set=[mb])\n        m.fit(A, y[i1], verbose=False, **kw)\n        oof[i2] = _link(_xgb_raw(m, B, mb))\n        tp += _link(_xgb_raw(m, C, mc)) / len(FOLDS)\n        it = int(fixed_iter) if fixed_iter else int(m.best_iteration)\n        iters.append(it)\n        print(f\'  fold {k}: #{it} {COMP[\'metric\']} {score(y[i2], oof[i2], COMP):.5f} {time.time() - T0:.0f}s\', flush=True)\n    save(oof, tp, note=note, params=P, iters=iters, n_features=A.shape[1], fixed_iter=fixed_iter, leak=leak)\n\ndef digest(a):\n    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()\n\ndef file_sha(path):\n    h = hashlib.sha256()\n    with open(path, \'rb\') as f:\n        for block in iter(lambda: f.read(1 << 20), b\'\'):\n            h.update(block)\n    return h.hexdigest()\n\ndef write_json(path, value):\n    path = Path(path)\n    tmp = path.with_suffix(\'.tmp\')\n    tmp.write_text(json.dumps(value, indent=2, default=str) + \'\\n\')\n    tmp.replace(path)\n\ndef run_controls(data=None, out=\'repeated_null_outputs\', device=None, benchmark=False):\n    global train, test, y, FOLDS, SPLIT_FP, T0, XGB_P\n    import shutil\n    data = Path(data) if data else next((Path(p) for p in [\'/kaggle/input/competitions/playground-series-s6e11\', \'/kaggle/input/playground-series-s6e11\', \'data\'] if (Path(p) / \'train.csv\').is_file()), None)\n    if data is None:\n        raise FileNotFoundError(\'Attach S6E11 train.csv and test.csv.\')\n    out = Path(out)\n    out.mkdir(parents=True, exist_ok=True)\n    device = device or (\'cuda\' if shutil.which(\'nvidia-smi\') else \'cpu\')\n    train = pd.read_csv(data / \'train.csv\')\n    test = pd.read_csv(data / \'test.csv\')\n    true_y = build_y(train, COMP)\n    all_folds = make_folds(true_y, COMP)\n    SPLIT_FP = fingerprint(all_folds, len(train))\n    assert SPLIT_FP == \'c7358c075703\'\n    assert len(train) == 699960 and len(test) == 299984 and (true_y.sum() == 11689)\n    fold_ids = np.empty(len(train), np.int8)\n    for k, (_, v) in enumerate(all_folds):\n        fold_ids[v] = k\n    np.save(out / \'frozen_fold_ids.npy\', fold_ids)\n    np.save(out / \'true_y.npy\', true_y)\n    environment = {\'utc\': datetime.now(timezone.utc).isoformat(), \'platform\': platform.platform(), \'packages\': {p: importlib.metadata.version(p) for p in [\'xgboost\', \'scikit-learn\', \'numpy\', \'pandas\', \'scipy\']}, \'device\': device, \'n_jobs\': 4, \'OMP_NUM_THREADS\': os.environ.get(\'OMP_NUM_THREADS\'), \'fold_md5_12\': SPLIT_FP, \'fold_ids_sha256\': digest(fold_ids), \'true_y_sha256\': digest(true_y), \'train_id_sha256\': digest(train[COMP[\'id_col\']].to_numpy()), \'input_file_sha256\': {p.name: file_sha(p) for p in [data / \'train.csv\', data / \'test.csv\']}, \'run_source_sha256\': file_sha(__file__) if \'__file__\' in globals() else None, \'historical_device\': \'cuda\', \'bitwise_reproduction_claim\': False, \'benchmark_only\': benchmark}\n    if device == \'cuda\':\n        import subprocess\n        environment[\'nvidia_smi\'] = subprocess.check_output([\'nvidia-smi\'], text=True)\n    write_json(out / \'environment.json\', environment)\n    write_json(out / \'protocol.json\', PROTOCOL)\n    for p in [\'train.csv\', \'test.csv\']:\n        assert environment[\'input_file_sha256\'][p] == PROTOCOL[\'input_file_sha256\'][\'data/\' + p], \'Raw input differs\'\n    tr, te, feats, tec = build_matrix(train, test)\n    assert len(feats) == 85 and len(tec) == 33\n    write_json(out / \'feature_manifest.json\', {\'before_TE\': feats, \'TE_keys\': tec, \'post_TE_count\': 118, \'dtypes\': {c: str(tr[c].dtype) for c in tr}})\n    XGB_P = dict(PROTOCOL[\'params\'])\n    XGB_P[\'device\'] = device\n    assert \'early_stopping_rounds\' not in XGB_P\n    summary = {\'status\': \'running\', \'environment\': environment, \'arms\': []}\n    arms = [(s, False) for s in PROTOCOL[\'seeds\']] + [(11, True)]\n    if benchmark:\n        arms = arms[:1]\n    for seed, leak in arms:\n        name = (\'positive_control_seed_\' if leak else \'null_seed_\') + str(seed)\n        arm = out / name\n        arm.mkdir(exist_ok=True)\n        perm = np.random.RandomState(seed).permutation(len(train))\n        y = true_y[perm].copy()\n        train[TARGET] = train[TARGET].to_numpy()\n        train[TARGET] = true_y[perm]\n        assert np.array_equal(build_y(train, COMP), y) and y.sum() == true_y.sum()\n        FOLDS = all_folds[:1] if benchmark else all_folds\n        T0 = time.time()\n\n        def save_arm(oof, tp, **meta):\n            oof = np.asarray(oof, np.float64)\n            tp = np.asarray(tp, np.float64)\n            from sklearn.metrics import roc_auc_score\n            assert oof.shape == (len(train),) and tp.shape == (len(test),)\n            assert np.isfinite(oof).all() and np.isfinite(tp).all()\n            fs = [float(roc_auc_score(y[v], oof[v])) for _, v in FOLDS]\n            a = float(roc_auc_score(y, oof)) if not benchmark else None\n            np.save(arm / \'oof.npy\', oof)\n            np.save(arm / \'test.npy\', tp)\n            np.save(arm / \'shuffled_y.npy\', y)\n            meta.update(name=name, seed=seed, status=\'benchmark_one_fold\' if benchmark else \'complete\', pooled_auc=a, fold_auc=fs, fold_positive_counts=[int(y[v].sum()) for _, v in FOLDS], original_fold_fp=SPLIT_FP, fold_ids_sha256=digest(fold_ids), shuffled_y_sha256=digest(y), permutation_sha256=digest(perm), oof_sha256=file_sha(arm / \'oof.npy\'), test_sha256=file_sha(arm / \'test.npy\'), elapsed_seconds=time.time() - T0, model_seed=42, device=device, abs_deviation=None if a is None else abs(a - 0.5), old_gate_pass=None if a is None else bool(abs(a - 0.5) < 0.003), leakage=\'validation TE fit on A union B\' if leak else \'outer-fitting-label-only TE\')\n            write_json(arm / \'meta.json\', meta)\n            summary[\'arms\'].append(meta)\n            write_json(out / \'report.json\', summary)\n        globals()[\'save\'] = save_arm\n        print(\'START\', name, \'device\', device, \'positive counts\', [int(y[v].sum()) for _, v in FOLDS], flush=True)\n        run_xgb(tr, te, feats, tec, fixed_iter=1200, P={\'n_jobs\': 4, \'device\': device}, leak=leak, note=\'H4 predeclared \' + name)\n        gc.collect()\n    summary[\'status\'] = \'benchmark_complete\' if benchmark else \'complete\'\n    summary[\'interpretation\'] = \'Five full refits describe repeated null behavior; no 5% tail calibration, no automatic gate/candidate change. Positive control assesses one intentional validation-TE leak only.\'\n    if not benchmark:\n        vals = np.array([a[\'pooled_auc\'] for a in summary[\'arms\'] if not a[\'leak\']])\n        summary[\'descriptive_null_summary\'] = {\'mean\': float(vals.mean()), \'sample_sd\': float(vals.std(ddof=1)), \'min\': float(vals.min()), \'max\': float(vals.max()), \'old_gate_exceedances\': int((np.abs(vals - 0.5) >= 0.003).sum()), \'n\': len(vals)}\n    write_json(out / \'report.json\', summary)\n    print(json.dumps(summary, indent=2), flush=True)\n    return summary\nif __name__ == \'__main__\' and RUN_REPEATED_NULL:\n    repeated_null_results = run_controls()\n'}
for filename, source in reproduction_sources.items():
    (output/filename).write_text(source)

# %% cell 50
"""Summarise complete recorded refits; these tables do not refit the pipelines."""
recorded_null_tables = {}
for key, pipeline in recorded_controls['pipelines'].items():
    print(pipeline['label'])
    frame = pd.DataFrame(pipeline['arms'])
    frame['outside band'] = (frame['auc']-.5).abs() >= .003
    table = frame[['seed', 'control', 'auc', 'outside band']].rename(columns={'auc':'OOF AUC'})
    table = table.set_index(['seed', 'control'])
    assert table.index.is_unique
    display(styled_table(table, {'OOF AUC':'{:.7f}'}))
    honest = frame.loc[frame.control.eq('shuffle')].copy()
    assert len(honest) == 5 and honest.seed.tolist() == [11, 23, 47, 101, 211]
    recorded_null_tables[key] = honest
    print(f"The five predeclared null refits range from {honest.auc.min():.7f} "
          f"to {honest.auc.max():.7f}; {int(honest['outside band'].sum())} of 5 "
          'are outside the unchanged diagnostic band.')
    print('This count describes these five refits; it does not estimate a false-positive rate.')
(output/'recorded_repeated_controls.json').write_text(json.dumps(recorded_controls, indent=2))
print('The recorded results and optional reproduction scripts are saved with the notebook outputs.')

# %% cell 51
"""Display the two diagnostic protocols separately on a common AUC scale."""
all_null = np.concatenate([frame.auc.to_numpy() for frame in recorded_null_tables.values()])
low, high = min(float(all_null.min()), .497), max(float(all_null.max()), .503)
padding = max((high-low)*.12, .001)
fig, axes = plt.subplots(2, 1, figsize=(7.3, 7.6), sharex=True)
titles = {'frequency':'Frequency encoding: folds rebuilt after shuffling',
          'xgboost':'Historical XGBoost: original folds held fixed'}
for ax, (key, frame) in zip(axes, recorded_null_tables.items()):
    ax.axvspan(.497, .503, color='#E8EDF1', zorder=0)
    ax.axvline(.5, color='#52616E', linestyle='--', linewidth=1)
    ax.scatter(frame.auc, np.arange(len(frame)), color='#0072B2', s=55, zorder=3)
    ax.set_yticks(np.arange(len(frame)), [f'Seed {seed}' for seed in frame.seed])
    ax.invert_yaxis()
    ax.set_ylim(4.6, -.6)
    ax.set_xlim(low-padding, high+padding)
    ax.set_title(titles[key], loc='left', fontsize=14, pad=16)
    ax.grid(axis='x')
    ax.tick_params(axis='x', labelbottom=True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
axes[-1].set_xlabel('Shuffled-label OOF AUC')
fig.subplots_adjust(left=.18, right=.97, top=.91, bottom=.12, hspace=.65)
show_figure(fig,
    'Each point is the pooled AUC from a complete five-fold refit on one label permutation. '
    'The shaded 0.497 to 0.503 band is the existing diagnostic rule, not a confidence interval. '
    'The deliberately contaminated controls are reported in the tables. '
    'The protocols differ in preprocessing, fold assignments and device.',
    'repeated_null_controls.png')

# %% cell 53
"""Optional reproduction runs in separate modules with their own output folders."""
import importlib.util

def load_reproducer(filename):
    path = output/filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if CFG.get('run_repeated_frequency', False):
    frequency_module = load_reproducer('run_frequency_controls.py')
    recomputed_frequency = frequency_module.run_repeated_frequency(DATA, output/'recomputed_frequency')
else:
    print('The frequency table is a recorded snapshot. Its six-arm reproduction is disabled.')
if CFG.get('run_repeated_h4', False):
    h4_module = load_reproducer('run_h4_controls.py')
    recomputed_h4 = h4_module.run_controls(data=DATA, out=output/'recomputed_h4', device='cuda')
else:
    print('The historical XGBoost table is a recorded GPU snapshot. Its six-arm reproduction is disabled.')
print('Optional refits use the original recorded protocols, independently of edited baseline settings.')

# %% cell 54
import base64 as s15_h5_b64, zlib as s15_h5_zlib
h5_payload=json.loads(s15_h5_zlib.decompress(s15_h5_b64.b64decode('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')))
recorded_h5_study=h5_payload["record"]
h5_reproduction_sources=h5_payload["sources"]
for filename, source in h5_reproduction_sources.items():
    (output/filename).write_text(source)

# %% cell 55
def h5_gate_figure(study):
    fig,ax=plt.subplots(figsize=(7.3,4.0))
    fig.subplots_adjust(left=.36,right=.96,top=.82,bottom=.22)
    values=[.5,study['raw_flag']['auc_no_fit'],study['h5']['control_auc']]
    labels=['Constant score','Raw new-merchant flag','H5 shuffled-label model']
    ax.axvspan(.497,.503,color='#E8EDF1',zorder=0)
    ax.axvline(.5,color='#52616E',linestyle='--',linewidth=1)
    for row,(value,colour) in enumerate(zip(values,['#52616E','#0072B2','#D55E00'])):
        ax.scatter(value,row,s=60,color=colour,zorder=3)
        ax.annotate(f'{value:.7f}',(value,row),xytext=(9,0),textcoords='offset points',va='center',fontsize=14)
    ax.set_yticks(range(3),labels);ax.set_ylim(2.6,-.6);ax.set_xlim(.496,.510)
    ax.set(title='One shuffled label vector, three scores',xlabel='AUC on the seed-0 shuffled labels')
    ax.xaxis.set_major_locator(plt.MaxNLocator(4));ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x:.3f}'))
    ax.grid(axis='x')
    return fig


fig=h5_gate_figure(recorded_h5_study)
fig.savefig('h5_gate_snapshot.png',dpi=100,facecolor='white');plt.close(fig)

# %% cell 57
study = recorded_h5_study
recipe_table = pd.DataFrame({
    'Reference':[study['h5']['baseline_auc'],0.0,study['d1']['seed0_reproductions']['baseline']['new_auc']],
    'H5':[study['h5']['auc'],study['h5']['delta'],study['h5']['control_auc']]},
    index=['OOF AUC','Delta vs reference','First shuffled AUC'])
display(styled_table(recipe_table,{'Reference':'{:.7f}','H5':'{:.7f}'}))
distance = abs(study['h5']['control_auc']-.5)
display(callout(f"The original H5 control is {distance:.7f} from 0.5, beyond the old 0.003 filter. "
                "This reproduces its recorded failure. The filter was an operational rule without a calibrated significance level."))

# %% cell 60
"""Optional H5/H6 reproduction. Importing this module performs no fits or submissions.

Call run_candidates(data, baseline_output, output_directory). The baseline folder
must contain oof.npy, folds.csv and metrics.json from the default five-fold run:
300 trees, 31 leaves, learning_rate=.05 and all 24 raw predictors. Presentation and
optional flags may vary.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time,importlib.metadata
import numpy as np,pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

def experiment_matrices(fitting_frame, other_frames, columns):
    cats = [c for c in columns if not pd.api.types.is_numeric_dtype(fitting_frame[c])]
    nums = [c for c in columns if c not in cats]
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if cats:
        enc.fit(fitting_frame[cats].astype('string').fillna('<missing>').astype(str))
    result = []
    for frame in [fitting_frame] + other_frames:
        values = frame[nums].to_numpy(dtype=np.float32)
        if cats:
            codes = enc.transform(frame[cats].astype('string').fillna('<missing>').astype(str))
            values = np.column_stack([values, codes]).astype(np.float32)
        result.append(pd.DataFrame(values, columns=nums+cats, index=frame.index))
    return result, list(range(len(nums), len(columns)))

PARAMS=dict(n_estimators=600,num_leaves=15,learning_rate=.03,random_state=42,n_jobs=4,
    min_child_samples=40,reg_lambda=12,subsample=.85,subsample_freq=1,colsample_bytree=.85,
    verbosity=-1,deterministic=True,force_col_wise=True)
FLAGS=['is_foreign_transaction','is_new_merchant','used_vpn','ip_country_mismatch',
       'billing_shipping_mismatch','is_ai_generated_scam_attempt']

def run_candidates(data,baseline_output,output_directory):
    data=Path(data);baseline_output=Path(baseline_output);output_directory=Path(output_directory)
    output_directory.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(data/'train.csv');test=pd.read_csv(data/'test.csv');sample=pd.read_csv(data/'sample_submission.csv')
    y=train.is_fraud.to_numpy();features=[c for c in test if c!='id']
    assert len(train)==699960 and len(test)==299984 and y.sum()==11689 and len(features)==24
    # The reference must be the default native300/31 baseline, not an edited fork.
    baseline_metrics=json.loads((baseline_output/'metrics.json').read_text())
    config=baseline_metrics['config']
    required_config={'competition':'playground-series-s6e11','task':'binary','metric':'auc',
        'target':'is_fraud','id_col':'id','positive_class':1,'split':'iid',
        'n_estimators':300,'num_leaves':31,'learning_rate':.05,'seed':42,'n_splits':5}
    for key,value in required_config.items():
        if config.get(key)!=value:
            raise ValueError(f'Default baseline required: {key} must be {value!r}, got {config.get(key)!r}.')
    if config.get('drop_features',[])!=[] or baseline_metrics.get('features')!=features:
        raise ValueError('Default baseline must use exactly the same ordered 24 raw predictors, with no dropped features.')
    reference=np.load(baseline_output/'oof.npy')
    if reference.ndim==2:
        if reference.shape!=(len(train),2):
            raise ValueError('Default binary baseline must have two probability columns.')
        reference=reference[:,1]
    assert reference.shape==y.shape
    if not np.isfinite(reference).all() or not ((reference>=0)&(reference<=1)).all():
        raise ValueError('Baseline OOF must contain finite probabilities in [0,1].')
    reference_auc=float(roc_auc_score(y,reference))
    if abs(reference_auc-float(baseline_metrics['pooled_oof']))>1e-12:
        raise ValueError('Baseline OOF and metrics do not describe these training labels.')
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(train,y))
    ids=np.full(len(train),-1,np.int16)
    for k,(_,v) in enumerate(folds):ids[v]=k
    saved=pd.read_csv(baseline_output/'folds.csv')
    np.testing.assert_array_equal(saved.id,train.id);np.testing.assert_array_equal(saved.fold,ids)
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'params':PARAMS,'flags':FLAGS,
        'scope':'Fixed original H5/H6 protocols. Paired OOF gains are exploratory; no automatic admission or submission.',
        'input_sha256':{p:hashlib.sha256((data/p).read_bytes()).hexdigest() for p in ['train.csv','test.csv']},
        'libraries':{p:importlib.metadata.version(p) for p in ['lightgbm','numpy','pandas','scikit-learn']}}
    (output_directory/'protocol.json').write_text(json.dumps(protocol,indent=2))
    report={}
    for name in ['h5','h6']:
        tr,te=train,test
        columns=features
        if name=='h6':
            tr,te=train.copy(),test.copy()
            for frame in [tr,te]:
                assert not frame[FLAGS].isna().any().any() and frame[FLAGS].isin([True,False]).all().all()
                frame['supplied_flag_count']=frame[FLAGS].astype(np.uint8).sum(axis=1).astype(np.uint8)
            columns=features+['supplied_flag_count']
        destination=output_directory/name;destination.mkdir(exist_ok=True)
        pred=np.zeros(len(test));oof=np.full(len(train),np.nan);deltas=[]
        compare=reference if name=='h5' else report['h5']['oof']
        for k,(fit,valid) in enumerate(folds):
            (a,b,c),cats=experiment_matrices(tr.iloc[fit],[tr.iloc[valid],te],columns)
            model=LGBMClassifier(**PARAMS).fit(a,y[fit],categorical_feature=cats)
            oof[valid]=model.predict_proba(b)[:,1];pred+=model.predict_proba(c)[:,1]/5
            deltas.append(float(roc_auc_score(y[valid],oof[valid])-roc_auc_score(y[valid],compare[valid])))
            print(name,'true fold',k,'delta',deltas[-1],flush=True)
        assert np.isfinite(oof).all() and np.isfinite(pred).all()
        np.save(destination/'oof.npy',oof);np.save(destination/'test.npy',pred)
        pd.DataFrame({'id':train.id,'fold':ids}).to_csv(destination/'folds.csv',index=False)
        labels=np.random.RandomState(0).permutation(y);control=np.full(len(y),np.nan);control_ids=np.full(len(y),-1,np.int16);scores=[]
        for k,(fit,valid) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,labels)):
            (a,b),cats=experiment_matrices(tr.iloc[fit],[tr.iloc[valid]],columns)
            model=LGBMClassifier(**PARAMS).fit(a,labels[fit],categorical_feature=cats)
            control[valid]=model.predict_proba(b)[:,1];control_ids[valid]=k
            scores.append(float(roc_auc_score(labels[valid],control[valid])))
            print(name,'null fold',k,'AUC',scores[-1],flush=True)
        assert np.isfinite(control).all()
        np.savez_compressed(destination/'control.npz',oof=control,labels=labels,fold_ids=control_ids)
        auc=float(roc_auc_score(y,oof));null_auc=float(roc_auc_score(labels,control))
        measured={'auc':auc,'reference_auc':float(roc_auc_score(y,compare)),
            'delta':auc-float(roc_auc_score(y,compare)),'fold_deltas':deltas,
            'control_auc':null_auc,'control_fold_auc':scores,'old_single_control_pass':bool(abs(null_auc-.5)<.003),
            'disposition':'Research output only; no automatic admission or submission'}
        (destination/'report.json').write_text(json.dumps(measured,indent=2))
        order=pd.Index(test.id).get_indexer(sample.id);assert (order>=0).all()
        candidate=sample.copy();candidate.is_fraud=pred[order];candidate.to_csv(destination/'submission.csv',index=False)
        report[name]={**measured,'oof':oof}
    return {name:{k:v for k,v in value.items() if k!='oof'} for name,value in report.items()}

# %% cell 62
raw_auc = roc_auc_score(np.random.RandomState(0).permutation(train[target].to_numpy()),
                        train['is_new_merchant'].astype(int))
print(f'The fixed raw new-merchant flag scores {raw_auc:.7f} on the seed-0 shuffled labels, without fitting.')
assert abs(raw_auc-study['raw_flag']['auc_no_fit']) < 1e-12
audit_rows = [
    ['Saved labels and fold assignments', 'Reconstructed exactly'],
    ['CSV round trip and native matrices', 'Identical model inputs'],
    ['Held-out target-column poisoning', 'Ordinary matrices unchanged; copied-target canary detected'],
    ['AUC after ranking within each fold', f"{study['no_fit_audit']['within_fold_rank_normalised_auc']:.7f}"],
    ['Raw new-merchant flag without fitting', f'{raw_auc:.7f}']]
display(styled_table(pd.DataFrame(audit_rows,columns=['check','recorded result']).set_index('check'))
        .set_properties(**{'white-space':'normal','overflow-wrap':'anywhere'}))

# %% cell 64
paired = pd.DataFrame(study['d1']['paired_new_seed_results'])
display(styled_table(paired[['seed','baseline_auc','h5_auc','h5_minus_baseline']].set_index('seed')
        .rename(columns={'baseline_auc':'Reference','h5_auc':'H5','h5_minus_baseline':'H5 minus reference'}),
        {'Reference':'{:.7f}','H5':'{:.7f}','H5 minus reference':'{:+.7f}'}))
summary = study['d1']['descriptive_summary']
for key,label in [('baseline','Reference'),('h5','H5')]:
    row=summary[key]
    display(callout(f"{label}: the 10 prescribed null AUCs range from {row['min']:.7f} to {row['max']:.7f}. "
                    f"The old filter excludes {row['outside_old_band_count']} of these 10 runs. "
                    "These are descriptive counts, not an estimated false-positive rate."))
holdout_summary = {}
for row in study['d1']['holdouts']:
    holdout_summary[f"Seed {row['seed']}"]={
        'Held-out rows':f"{row['rows']:,}",'Positive labels':f"{row['positives']:,}",
        'AUC':f"{row['auc']:.7f}",'Conditional MC p':f"{row['conditional_MC_p_two_sided']:.4f}",
        'Bonferroni p (2)':f"{row['Bonferroni_2_p']:.4f}"}
display(styled_table(pd.DataFrame(holdout_summary)))

# %% cell 65
fig,ax=plt.subplots(figsize=(7.3,6.0))
rows=pd.DataFrame(recorded_h5_study['d1']['paired_new_seed_results'])
for k,row in rows.iterrows():
    ax.plot([row.baseline_auc,row.h5_auc],[k,k],color='#C5CDD3',linewidth=1,zorder=1)
ax.axvspan(.497,.503,color='#E8EDF1',zorder=0)
ax.axvline(.5,color='#52616E',linestyle='--',linewidth=1)
ax.scatter(rows.baseline_auc,np.arange(len(rows)),color='#0072B2',s=45,label='300-tree reference',zorder=3)
ax.scatter(rows.h5_auc,np.arange(len(rows)),color='#D55E00',marker='D',s=45,label='H5 recipe',zorder=3)
values=np.r_[rows.baseline_auc,rows.h5_auc,.497,.503]
ax.set_xlim(values.min()-.001,values.max()+.001)
ax.set_yticks(np.arange(len(rows)),[str(int(seed)) for seed in rows.seed]);ax.set_ylim(len(rows)-.4,-.6)
ax.set(title='Complete null refits on 10 new permutations',xlabel='Pooled shuffled-label OOF AUC',ylabel='Permutation seed')
ax.xaxis.set_major_locator(plt.MaxNLocator(5));ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x:.3f}'))
ax.grid(axis='x');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2,fontsize=14)
fig.subplots_adjust(left=.16,right=.97,top=.88,bottom=.22)
show_figure(fig,
    'Each pair uses the same shuffled labels and rebuilt folds for the reference and H5. '
    'Every point summarises a complete five-fold refit. The shaded band is the old operational rule, '
    'not a confidence interval. These 10 permutations do not calibrate a 5% tail or establish a false-positive rate.',
    'paired_null_diagnosis.png')

# %% cell 67
probe = pd.DataFrame(study['d2']['arms'])
probe['arm'] = probe.name.map({'balanced_overlap':'Overlap 195','edit_count_matched':'Overlap 239'})
display(styled_table(probe[['seed','arm','relabel_only_auc','combined_auc']].set_index(['seed','arm'])
        .rename(columns={'relabel_only_auc':'Original score / edited labels','combined_auc':'Refit score / edited labels'}),
        {'Original score / edited labels':'{:.7f}','Refit score / edited labels':'{:.7f}'}))

# %% cell 70
feature_comparison=pd.DataFrame({'value':[
    f"{study['h5']['auc']:.7f}",f"{study['h6']['auc']:.7f}",
    f"{study['h6']['delta_vs_h5']:+.7f}",f"{study['h6']['control_auc']:.7f}"]},
    index=['H5 OOF AUC','H6 OOF AUC','H6 minus H5','H6 shuffled AUC'])
display(styled_table(feature_comparison))
display(styled_table(pd.DataFrame({'H5 minus reference':study['h5']['fold_deltas'],
                                  'H6 minus H5':study['h6']['fold_deltas']}).rename_axis('fold'),
                    {'H5 minus reference':'{:+.7f}','H6 minus H5':'{:+.7f}'}))

# %% cell 71
fig,ax=plt.subplots(figsize=(7.3,3.8))
values=[recorded_h5_study['h5']['baseline_auc'],recorded_h5_study['h5']['auc'],recorded_h5_study['h6']['auc']]
names=['300-tree reference','H5 regularised recipe','H6 adds a flag count']
for row,(value,colour) in enumerate(zip(values,['#52616E','#0072B2','#D55E00'])):
    ax.scatter(value,row,s=60,color=colour,zorder=3)
    ax.annotate(f'{value:.7f}',(value,row),xytext=(7,0),textcoords='offset points',va='center',fontsize=14)
ax.set_yticks(range(3),names);ax.set_ylim(2.6,-.6)
ax.set_xlim(min(values)-.0005,max(values)+.0007)
ax.set(title='Recorded recipe scores on the same folds',xlabel='Pooled true-label OOF AUC')
ax.xaxis.set_major_locator(plt.MaxNLocator(3));ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x:.4f}'))
ax.grid(axis='x');fig.subplots_adjust(left=.32,right=.97,top=.82,bottom=.24)
show_figure(fig,
    'All three scores use the same five true-label folds. H5 changes a complete training recipe. '
    'H6 adds one row-local flag count to H5 and also changes column-subsampling behaviour. '
    'Its small extra gain did not meet the predeclared selection criterion. '
    'OOF scores are exploratory development measurements, not private-leaderboard guarantees.',
    'recorded_recipe_scores.png')

# %% cell 75
v2=study['v2']
true_comparison=pd.DataFrame(v2['true_comparisons'])
true_comparison['partition']=['A','B']
display(styled_table(true_comparison[['partition','baseline','h5','delta']].set_index('partition')
      .rename(columns={'baseline':'Reference AUC','h5':'H5 AUC','delta':'H5 minus reference'}),
      {'Reference AUC':'{:.7f}','H5 AUC':'{:.7f}','H5 minus reference':'{:+.7f}'}))
for split_seed,label in zip(study['v2_protocol']['splits'],['A','B']):
    display(callout(f'Partition {label}: the following three negative controls form part of the single six-test family.'))
    rows=pd.DataFrame([r for r in v2['negative_controls'] if r['split_seed']==split_seed])
    display(styled_table(rows[['label_seed','auc','p','bonferroni_p']].set_index('label_seed')
          .rename(columns={'auc':'AUC','p':'Raw MC p','bonferroni_p':'Bonferroni p (6)'}),
          {'AUC':'{:.7f}','Raw MC p':'{:.4f}','Bonferroni p (6)':'{:.4f}'}))
canaries=pd.DataFrame(v2['canaries'])
canaries['partition']=['A','B']
display(styled_table(canaries[['partition','auc','p','bonferroni_p']].set_index('partition')
      .rename(columns={'auc':'Copied-target AUC','p':'Raw MC p','bonferroni_p':'Bonferroni p (2)'}),
      {'Copied-target AUC':'{:.7f}','Raw MC p':'{:.4f}','Bonferroni p (2)':'{:.4f}'}))
(output/'recorded_h5_studies.json').write_text(json.dumps(study,indent=2));

# %% cell 77
if CFG.get('run_candidate_studies', False):
    module = load_reproducer('run_candidate_studies.py')
    candidate_replay = module.run_candidates(DATA, output, output/'recomputed_candidates')
    print(candidate_replay)
if CFG.get('run_h5_diagnosis', False):
    module = load_reproducer('run_h5_diagnosis.py')
    module.run('all', data=DATA, out=output/'recomputed_h5_diagnosis')
if CFG.get('run_overlap_probe', False):
    control_path = output/'recomputed_candidates'/'h5'/'control.npz'
    if not control_path.exists():
        raise FileNotFoundError('Run candidate studies first to create the H5 control archive.')
    module = load_reproducer('run_overlap_probe.py')
    module.run_overlap_probe(DATA, control_path, output/'recomputed_overlap_probe')
if CFG.get('run_prospective_audit', False):
    module = load_reproducer('run_prospective_audit.py')
    module.run_prospective_audit(DATA, output/'recomputed_prospective_audit')
print('Recorded investigations are displayed above. Optional reruns write separate reports; no file is uploaded.')

# %% cell 80
recorded_digit_study = {'publication_ready': True,
 'stage': 'Value and digit encodings, recorded 2 November 2026 local time',
 'protocol': {'created_utc': '2026-11-01T20:24:03.210438+00:00',
              'study': 'Prospective S6E11 value and digit ablations; no candidate fit yet',
              'input_sha256': {'train.csv': '87eaa10b8d17a7e4f657b8da1c66242431338bdd8b6ad45861521ac0c0e9fbd5',
                               'test.csv': '10f516959020357b565abaaa4fd25baae97fc2f536d0805695bb7cc15c1660f0',
                               'sample_submission.csv': '13fc6ed8b73de5329884ec490f2a0b750cf29a7cda772b78cfa0cf486f6f35fa'},
              'features': ['amount_usd',
                           'hours_since_last_txn',
                           'txn_count_last_24h',
                           'distance_from_home_km',
                           'card_age_months',
                           'customer_age',
                           'account_balance_usd',
                           'cvv_retry_count',
                           'velocity_score',
                           'time_of_day_hour',
                           'day_of_week',
                           'merchant_risk_score',
                           'prior_disputes',
                           'merchant_category',
                           'card_type',
                           'auth_method',
                           'channel',
                           'device_type',
                           'is_foreign_transaction',
                           'is_new_merchant',
                           'used_vpn',
                           'ip_country_mismatch',
                           'billing_shipping_mismatch',
                           'is_ai_generated_scam_attempt'],
              'reference': {'name': 'H5',
                            'auc': 0.9464870105109693,
                            'oof_sha256': 'f94d5389f99524f8f901411b9f040f73bee822dd0b46b518c598d7e722b14d1e',
                            'fold_ids_sha256': '946567353f68c167ea4f9aa037aad048769d78f4dcc0069791cd3ca0731eca45'},
              'h5_params': {'n_estimators': 600,
                            'num_leaves': 15,
                            'learning_rate': 0.03,
                            'random_state': 42,
                            'n_jobs': 4,
                            'min_child_samples': 40,
                            'reg_lambda': 12,
                            'subsample': 0.85,
                            'subsample_freq': 1,
                            'colsample_bytree': 0.85,
                            'verbosity': -1,
                            'deterministic': True,
                            'force_col_wise': True},
              'recipes': {'h7_cents': {'view': 'cents',
                                       'params': {'n_estimators': 600,
                                                  'num_leaves': 15,
                                                  'learning_rate': 0.03,
                                                  'random_state': 42,
                                                  'n_jobs': 4,
                                                  'min_child_samples': 40,
                                                  'reg_lambda': 12,
                                                  'subsample': 0.85,
                                                  'subsample_freq': 1,
                                                  'colsample_bytree': 0.85,
                                                  'verbosity': -1,
                                                  'deterministic': True,
                                                  'force_col_wise': True},
                                       'feature_count': 30},
                          'h8_per_value': {'view': 'per_value',
                                           'params': {'n_estimators': 600,
                                                      'num_leaves': 15,
                                                      'learning_rate': 0.03,
                                                      'random_state': 42,
                                                      'n_jobs': 4,
                                                      'min_child_samples': 40,
                                                      'reg_lambda': 12,
                                                      'subsample': 0.85,
                                                      'subsample_freq': 1,
                                                      'colsample_bytree': 0.85,
                                                      'verbosity': -1,
                                                      'deterministic': True,
                                                      'force_col_wise': True},
                                           'feature_count': 48},
                          'h9_digits': {'view': 'digits',
                                        'params': {'n_estimators': 600,
                                                   'num_leaves': 15,
                                                   'learning_rate': 0.03,
                                                   'random_state': 42,
                                                   'n_jobs': 4,
                                                   'min_child_samples': 40,
                                                   'reg_lambda': 12,
                                                   'subsample': 0.85,
                                                   'subsample_freq': 1,
                                                   'colsample_bytree': 0.85,
                                                   'verbosity': -1,
                                                   'deterministic': True,
                                                   'force_col_wise': True},
                                        'feature_count': 96},
                          'h10_small_trees': {'view': 'digits',
                                              'params': {'n_estimators': 1600,
                                                         'num_leaves': 4,
                                                         'learning_rate': 0.03,
                                                         'random_state': 42,
                                                         'n_jobs': 4,
                                                         'min_child_samples': 100,
                                                         'reg_lambda': 1.0,
                                                         'subsample': 0.8,
                                                         'subsample_freq': 1,
                                                         'colsample_bytree': 0.5,
                                                         'verbosity': -1,
                                                         'deterministic': True,
                                                         'force_col_wise': True},
                                              'feature_count': 96}},
              'digit_definition': 'Six nonnegative float columns. k=np.rint(100*x), '
                                  'cent=k%100,integer=k//100,tens=k//1000,tenths=k//10,leading1/leading2/length '
                                  'of integer; zero has leading0,length1. Quantisation '
                                  'affects derived keys ONLY. The test exception id962974 '
                                  'amount9.982 remains unchanged in raw inputs/raw-value TE; '
                                  'record it, never infer its label.',
              'encoding': 'Official sklearn1.6.1 TargetEncoder '
                          'binary/smooth20/cv5/shuffleTrue/RS0; per-column fit_transform on '
                          'outer-fitting keys and labels for inner cross-fitting; transform '
                          'outervalidation/test using allouterfitting labels. No source rows, '
                          'H6 flagcount, outer eval_set/early stopping, outside predictions, '
                          'or ID/target inputs.',
              'fixed_cv': 'Original true-label SKFold5 shuffle42; exact H5foldIDs. Complete '
                          'RandomState0 label permutation independently re-stratified5x42 for '
                          'each same-budget negative pipeline. H9/H10 share identical '
                          'prepared matrices on each fold.',
              'cv_fit_budget': 40,
              'selection': 'Evaluate all4 recipes and all4 full seed0 controls. Eligible only '
                           'gainH5>=.0005, >=4of5positive pairedfold deltas, and unchanged '
                           'oldabs(nullAUC-.5)<.003. Choose highesteligiblepooledOOFonce '
                           '(ties favour earlierlisted recipe). If noneeligible, no '
                           'holdoutaudit/no submission. Selection is exploratory on '
                           'reuseddata, not unbiased performance.',
              'final_audit': {'splits': [20261105, 20261106],
                              'label_seeds': [2027, 2137, 2267],
                              'mc_seeds': {'20261105_2027': 202611051,
                                           '20261105_2137': 202611052,
                                           '20261105_2267': 202611053,
                                           '20261106_2027': 202611061,
                                           '20261106_2137': 202611062,
                                           '20261106_2267': 202611063},
                              'canary_mc_seeds': {'20261105': 202611059,
                                                  '20261106': 202611069},
                              'fit_budget': 12,
                              'criteria': 'Bothcandidate-minusH5trueholdoutgains>=.0005; '
                                          'all6Bonferroni-adjustednegativeMCp>=.05, '
                                          'both2test-adjustedcopiedtargetcanaryp<.01; '
                                          'inputtargetpoisonanddisjointnessaudits pass.',
                              'test': 'Singlefixedlabel-independent20%KFoldpartitioneachsplit. '
                                      'PCG64MC4999 fixedcountlabelassignments, absAUC-.5 with '
                                      'midrank ties/equalityextreme, p=(extreme+1)/5000. '
                                      'Bonferronifamilies6negative/2canary, '
                                      'dependenciesallowed. Alpha.05chosenbeforefits, '
                                      'strongerrejectionthresholdthan historicalV2.01; old '
                                      'fullCVbandalso required.',
                              'limits': 'Reused/overlappingcompetitiondata and '
                                        'modelselectionhistory. Not externalvalidation, '
                                        'pooledOOFp, calibratedweakleak detection, '
                                        'featureavailabilityaudit, or safetycertificate. '
                                        'Labels existinprocess; excludeheldlabelsfromfitAPI '
                                        'and savepredictionbeforemetrics. Canaries '
                                        'teststrongdirectcopy only.'},
              'submission_budget': 1,
              'submission_reading_rule': 'Bind exactlychosen5foldCV CSV hash before '
                                         'holdoutfits. Require '
                                         'numericpass+independentsciencereview+rootdecision. '
                                         'If auditfails do notfallback toanotherrecipe, '
                                         'changecriteria/seed/budget, or addmodels/blends. '
                                         'Atmost1acceptednewsubmission, '
                                         'reportpublicoutcomeeitherdirection againstH5.94709. '
                                         'No tuning/selection usingthe newpublicscore.',
              'history': 'OriginalH5FAIL and V2decision remain unchanged. Newstudy '
                         'retainsoldfullnullfilter and adds stronger '
                         'freshholdoutfamilythreshold; it does not retrospectively admit H6 '
                         'or H3/H4.',
              'credits': {'goodpjw2008': 'https://www.kaggle.com/code/goodpjw2008/s6e11-part-3-digits-leak-the-label-lb-0-95113',
                          'alexchilton': 'https://www.kaggle.com/code/alexchilton/credit-card-fraud-lb-0-9496-shallow-tree-stack',
                          'sklearn': 'https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.TargetEncoder.html'},
              'adaptation': 'Own bounded ablations with official inner-fold TE, '
                            'fixed1600trees/noouterES. Not exactreproductionofthe '
                            'publishedensembles or evidenceoftheirclaimedgeneratormechanism.',
              'field_sources_sha256': {'checks/stage10/field_digits/s6e11-part-3-digits-leak-the-label-lb-0-95113.ipynb': 'e498b85f828f1979643908cbd33df7f64f9e23a4ff1c7b47d655ff4e5db9f28a',
                                       'checks/stage10/field_eda/s6e11-deep-eda-original-data-artifacts.ipynb': '550117bec14d075b41fad0b6f07fc2a298eb2c6e53b35faec5a24fb6d9ad2291',
                                       'checks/stage10/field_stack/credit-card-fraud-lb-0-9496-shallow-tree-stack.ipynb': 'bd1ef11fc60c7de08ce31ea95e1d3e41ca128a9dae7c6e13e6ae3c966cdef380'},
              'runner_sha256': '7db01282d41372109e0da87e8abb70f166f5dc3762c492ced219a2a2e3ae2577',
              'resource_policy': '8GB localhost: prepare cents, per_value and digits '
                                 'viewgroups sequentially; H9/H10 reuse exactly the same '
                                 'digit matrices. FourLightGBMthreads, no '
                                 'parallelmodelworkers. Drop group matrices before nextview.',
              'additional_analyses': 'Save six outer-heldout raw-value target encodings from '
                                     'H8 for single-feature AUC plots against the signed raw '
                                     'magnitude AUC. Allsixspecifiedfloats are reported; no '
                                     'max(AUC,1-AUC), selecteddirection, newmodel or '
                                     'outcome-guided featurechoice. '
                                     'Quantization/non-centexceptions/unseenkeycounts are '
                                     'descriptive target-free input audits.'},
 'screen': {'status': 'complete',
            'started_utc': '2026-11-01T20:29:55.846402+00:00',
            'runner_sha256': '7db01282d41372109e0da87e8abb70f166f5dc3762c492ced219a2a2e3ae2577',
            'reference_auc': 0.9464870105109693,
            'libraries': {'lightgbm': '4.6.0',
                          'numpy': '2.2.6',
                          'pandas': '2.2.3',
                          'scikit-learn': '1.6.1',
                          'scipy': '1.18.0'},
            'recipes': {'h7_cents': {'true_folds': [{'fold': 0,
                                                     'auc': 0.9451728674877523,
                                                     'trees': 600,
                                                     'features': 30,
                                                     'seconds': 48.741409749985905,
                                                     'delta_vs_h5': 0.0003782253659612156},
                                                    {'fold': 1,
                                                     'auc': 0.9481143806548455,
                                                     'trees': 600,
                                                     'features': 30,
                                                     'seconds': 40.956476333987666,
                                                     'delta_vs_h5': -0.00017274998374006145},
                                                    {'fold': 2,
                                                     'auc': 0.9518239828022212,
                                                     'trees': 600,
                                                     'features': 30,
                                                     'seconds': 46.65528083301615,
                                                     'delta_vs_h5': 0.0006630819069388671},
                                                    {'fold': 3,
                                                     'auc': 0.9471281860249329,
                                                     'trees': 600,
                                                     'features': 30,
                                                     'seconds': 45.48120137499063,
                                                     'delta_vs_h5': 1.7207572529875925e-05},
                                                    {'fold': 4,
                                                     'auc': 0.9415367316795562,
                                                     'trees': 600,
                                                     'features': 30,
                                                     'seconds': 40.691937790979864,
                                                     'delta_vs_h5': 0.00019029623908095594}],
                                     'null_folds': [{'fold': 0,
                                                     'auc': 0.5056935499185289,
                                                     'trees': 600,
                                                     'features': 30,
                                                     'seconds': 29.768925083015347},
                                                    {'fold': 1,
                                                     'auc': 0.5056728904718557,
                                                     'trees': 600,
                                                     'features': 30,
                                                     'seconds': 30.20872049999889},
                                                    {'fold': 2,
                                                     'auc': 0.5107297075894642,
                                                     'trees': 600,
                                                     'features': 30,
                                                     'seconds': 31.499634750012774},
                                                    {'fold': 3,
                                                     'auc': 0.502284642382583,
                                                     'trees': 600,
                                                     'features': 30,
                                                     'seconds': 25.136768291005865},
                                                    {'fold': 4,
                                                     'auc': 0.5018511594566772,
                                                     'trees': 600,
                                                     'features': 30,
                                                     'seconds': 26.621390749991406}],
                                     'auc': 0.9467119892639174,
                                     'delta_vs_h5': 0.00022497875294813507,
                                     'control_auc': 0.5052004689704832,
                                     'positive_folds': 4,
                                     'fold_deltas': [0.0003782253659612156,
                                                     -0.00017274998374006145,
                                                     0.0006630819069388671,
                                                     1.7207572529875925e-05,
                                                     0.00019029623908095594],
                                     'old_null_pass': False,
                                     'eligible_for_final_audit': False,
                                     'submission_sha256': '62471528ed606bbffd4b35bb764a2651756cdc9d798642a4329c237cb38c6932'},
                        'h8_per_value': {'true_folds': [{'fold': 0,
                                                         'auc': 0.9463510251259609,
                                                         'trees': 600,
                                                         'features': 48,
                                                         'seconds': 54.11144887498813,
                                                         'delta_vs_h5': 0.0015563830041698257},
                                                        {'fold': 1,
                                                         'auc': 0.9496216216995531,
                                                         'trees': 600,
                                                         'features': 48,
                                                         'seconds': 55.38850020899554,
                                                         'delta_vs_h5': 0.001334491060967502},
                                                        {'fold': 2,
                                                         'auc': 0.9531933146921486,
                                                         'trees': 600,
                                                         'features': 48,
                                                         'seconds': 52.880937791022006,
                                                         'delta_vs_h5': 0.0020324137968662503},
                                                        {'fold': 3,
                                                         'auc': 0.9487017560784522,
                                                         'trees': 600,
                                                         'features': 48,
                                                         'seconds': 54.120016833010595,
                                                         'delta_vs_h5': 0.0015907776260492001},
                                                        {'fold': 4,
                                                         'auc': 0.9423239641404878,
                                                         'trees': 600,
                                                         'features': 48,
                                                         'seconds': 56.50397766599781,
                                                         'delta_vs_h5': 0.0009775287000125532}],
                                         'null_folds': [{'fold': 0,
                                                         'auc': 0.49663188873935504,
                                                         'trees': 600,
                                                         'features': 48,
                                                         'seconds': 39.10618129200884},
                                                        {'fold': 1,
                                                         'auc': 0.5017397949555849,
                                                         'trees': 600,
                                                         'features': 48,
                                                         'seconds': 43.291664458985906},
                                                        {'fold': 2,
                                                         'auc': 0.5002074199177037,
                                                         'trees': 600,
                                                         'features': 48,
                                                         'seconds': 39.511896750016604},
                                                        {'fold': 3,
                                                         'auc': 0.5024576564767719,
                                                         'trees': 600,
                                                         'features': 48,
                                                         'seconds': 38.7089896670077},
                                                        {'fold': 4,
                                                         'auc': 0.5024005185115759,
                                                         'trees': 600,
                                                         'features': 48,
                                                         'seconds': 42.665814834006596}],
                                         'auc': 0.9479860747258101,
                                         'delta_vs_h5': 0.0014990642148408062,
                                         'control_auc': 0.500608985938339,
                                         'positive_folds': 5,
                                         'fold_deltas': [0.0015563830041698257,
                                                         0.001334491060967502,
                                                         0.0020324137968662503,
                                                         0.0015907776260492001,
                                                         0.0009775287000125532],
                                         'old_null_pass': True,
                                         'eligible_for_final_audit': True,
                                         'submission_sha256': 'e40462419cdf5813af270089c78665a8a0131fc2ba78fb99652f83690c2f31b6'},
                        'h9_digits': {'true_folds': [{'fold': 0,
                                                      'auc': 0.9482684715298259,
                                                      'trees': 600,
                                                      'features': 96,
                                                      'seconds': 85.5125023330038,
                                                      'delta_vs_h5': 0.0034738294080347565},
                                                     {'fold': 1,
                                                      'auc': 0.9502944943361856,
                                                      'trees': 600,
                                                      'features': 96,
                                                      'seconds': 89.06737333300407,
                                                      'delta_vs_h5': 0.0020073636976000175},
                                                     {'fold': 2,
                                                      'auc': 0.9547848939710893,
                                                      'trees': 600,
                                                      'features': 96,
                                                      'seconds': 91.07966524999938,
                                                      'delta_vs_h5': 0.003623993075806964},
                                                     {'fold': 3,
                                                      'auc': 0.9505676156119874,
                                                      'trees': 600,
                                                      'features': 96,
                                                      'seconds': 93.11391362501308,
                                                      'delta_vs_h5': 0.0034566371595844414},
                                                     {'fold': 4,
                                                      'auc': 0.9432289525753708,
                                                      'trees': 600,
                                                      'features': 96,
                                                      'seconds': 80.743474124989,
                                                      'delta_vs_h5': 0.0018825171348955116}],
                                      'null_folds': [{'fold': 0,
                                                      'auc': 0.5017286010509148,
                                                      'trees': 600,
                                                      'features': 96,
                                                      'seconds': 66.70533545897342},
                                                     {'fold': 1,
                                                      'auc': 0.5014285143807145,
                                                      'trees': 600,
                                                      'features': 96,
                                                      'seconds': 74.04323700000532},
                                                     {'fold': 2,
                                                      'auc': 0.5026494379487292,
                                                      'trees': 600,
                                                      'features': 96,
                                                      'seconds': 68.23133395801415},
                                                     {'fold': 3,
                                                      'auc': 0.5045855042538996,
                                                      'trees': 600,
                                                      'features': 96,
                                                      'seconds': 58.32654629199533},
                                                     {'fold': 4,
                                                      'auc': 0.498072705268909,
                                                      'trees': 600,
                                                      'features': 96,
                                                      'seconds': 58.8947643339925}],
                                      'auc': 0.9493735374849556,
                                      'delta_vs_h5': 0.002886526973986281,
                                      'control_auc': 0.5017606892301936,
                                      'positive_folds': 5,
                                      'fold_deltas': [0.0034738294080347565,
                                                      0.0020073636976000175,
                                                      0.003623993075806964,
                                                      0.0034566371595844414,
                                                      0.0018825171348955116],
                                      'old_null_pass': True,
                                      'eligible_for_final_audit': True,
                                      'submission_sha256': '9405c5036ee8f94cf4b6e539beb43a71385c222812ce47b51df906fb719d3b7d'},
                        'h10_small_trees': {'true_folds': [{'fold': 0,
                                                            'auc': 0.94936233938769,
                                                            'trees': 1600,
                                                            'features': 96,
                                                            'seconds': 101.65111679199617,
                                                            'delta_vs_h5': 0.004567697265898896},
                                                           {'fold': 1,
                                                            'auc': 0.9507428544483091,
                                                            'trees': 1600,
                                                            'features': 96,
                                                            'seconds': 104.69645058401511,
                                                            'delta_vs_h5': 0.0024557238097234757},
                                                           {'fold': 2,
                                                            'auc': 0.9552754185395567,
                                                            'trees': 1600,
                                                            'features': 96,
                                                            'seconds': 104.17506191600114,
                                                            'delta_vs_h5': 0.004114517644274285},
                                                           {'fold': 3,
                                                            'auc': 0.9515709587779767,
                                                            'trees': 1600,
                                                            'features': 96,
                                                            'seconds': 102.42628612500266,
                                                            'delta_vs_h5': 0.004459980325573709},
                                                           {'fold': 4,
                                                            'auc': 0.9445551816400657,
                                                            'trees': 1600,
                                                            'features': 96,
                                                            'seconds': 96.51645512500545,
                                                            'delta_vs_h5': 0.00320874619959044}],
                                            'null_folds': [{'fold': 0,
                                                            'auc': 0.5020911378742665,
                                                            'trees': 1600,
                                                            'features': 96,
                                                            'seconds': 92.60956670800806},
                                                           {'fold': 1,
                                                            'auc': 0.500887954864531,
                                                            'trees': 1600,
                                                            'features': 96,
                                                            'seconds': 100.39043016699725},
                                                           {'fold': 2,
                                                            'auc': 0.5086066852655937,
                                                            'trees': 1600,
                                                            'features': 96,
                                                            'seconds': 83.02113895898219},
                                                           {'fold': 3,
                                                            'auc': 0.5022687118617521,
                                                            'trees': 1600,
                                                            'features': 96,
                                                            'seconds': 77.6317667500116},
                                                           {'fold': 4,
                                                            'auc': 0.5028767407224494,
                                                            'trees': 1600,
                                                            'features': 96,
                                                            'seconds': 77.14073779198225}],
                                            'auc': 0.9502590175536698,
                                            'delta_vs_h5': 0.0037720070427005314,
                                            'control_auc': 0.5034035799553033,
                                            'positive_folds': 5,
                                            'fold_deltas': [0.004567697265898896,
                                                            0.0024557238097234757,
                                                            0.004114517644274285,
                                                            0.004459980325573709,
                                                            0.00320874619959044],
                                            'old_null_pass': False,
                                            'eligible_for_final_audit': False,
                                            'submission_sha256': 'e70bc1c4818482b78de38e90b13a3e4d5693dd8e4b0faf2f5071b9d0c7bcd499'}},
            'univariate_numeric': [{'column': 'amount_usd',
                                    'raw_auc': 0.5448494958612227,
                                    'heldout_per_value_te_auc': 0.5465696918145085},
                                   {'column': 'hours_since_last_txn',
                                    'raw_auc': 0.4317779424786956,
                                    'heldout_per_value_te_auc': 0.5849897734527576},
                                   {'column': 'distance_from_home_km',
                                    'raw_auc': 0.5038733190708004,
                                    'heldout_per_value_te_auc': 0.5108501971422594},
                                   {'column': 'account_balance_usd',
                                    'raw_auc': 0.5062529019486234,
                                    'heldout_per_value_te_auc': 0.5357721086550941},
                                   {'column': 'velocity_score',
                                    'raw_auc': 0.7107927599976092,
                                    'heldout_per_value_te_auc': 0.7064406682879019},
                                   {'column': 'merchant_risk_score',
                                    'raw_auc': 0.7489359267576935,
                                    'heldout_per_value_te_auc': 0.7493992870383831}],
            'selected_for_final_audit': 'h9_digits',
            'submission_decision': 'No upload: final audit, independent review and a separate '
                                   'root decision are still required.',
            'completed_utc': '2026-11-01T21:18:29.968445+00:00'},
 'input_audit': [{'file': 'train',
                  'column': 'amount_usd',
                  'non_cent_aligned_rows': 0,
                  'max_cent_residual': 5.820766091346741e-11,
                  'exceptions': [],
                  'zero_count': 0},
                 {'file': 'train',
                  'column': 'hours_since_last_txn',
                  'non_cent_aligned_rows': 0,
                  'max_cent_residual': 4.547473508864641e-13,
                  'exceptions': [],
                  'zero_count': 0},
                 {'file': 'train',
                  'column': 'distance_from_home_km',
                  'non_cent_aligned_rows': 0,
                  'max_cent_residual': 4.547473508864641e-13,
                  'exceptions': [],
                  'zero_count': 12},
                 {'file': 'train',
                  'column': 'account_balance_usd',
                  'non_cent_aligned_rows': 0,
                  'max_cent_residual': 9.313225746154785e-10,
                  'exceptions': [],
                  'zero_count': 0},
                 {'file': 'train',
                  'column': 'velocity_score',
                  'non_cent_aligned_rows': 0,
                  'max_cent_residual': 9.094947017729282e-13,
                  'exceptions': [],
                  'zero_count': 48499},
                 {'file': 'train',
                  'column': 'merchant_risk_score',
                  'non_cent_aligned_rows': 0,
                  'max_cent_residual': 9.094947017729282e-13,
                  'exceptions': [],
                  'zero_count': 1631},
                 {'file': 'test',
                  'column': 'amount_usd',
                  'non_cent_aligned_rows': 1,
                  'max_cent_residual': 0.1999999999999318,
                  'exceptions': [{'id': 962974, 'raw': 9.982}],
                  'zero_count': 0},
                 {'file': 'test',
                  'column': 'hours_since_last_txn',
                  'non_cent_aligned_rows': 0,
                  'max_cent_residual': 4.547473508864641e-13,
                  'exceptions': [],
                  'zero_count': 0},
                 {'file': 'test',
                  'column': 'distance_from_home_km',
                  'non_cent_aligned_rows': 0,
                  'max_cent_residual': 4.547473508864641e-13,
                  'exceptions': [],
                  'zero_count': 4},
                 {'file': 'test',
                  'column': 'account_balance_usd',
                  'non_cent_aligned_rows': 0,
                  'max_cent_residual': 9.313225746154785e-10,
                  'exceptions': [],
                  'zero_count': 0},
                 {'file': 'test',
                  'column': 'velocity_score',
                  'non_cent_aligned_rows': 0,
                  'max_cent_residual': 9.094947017729282e-13,
                  'exceptions': [],
                  'zero_count': 20927},
                 {'file': 'test',
                  'column': 'merchant_risk_score',
                  'non_cent_aligned_rows': 0,
                  'max_cent_residual': 9.094947017729282e-13,
                  'exceptions': [],
                  'zero_count': 671}],
 'audit': {'status': 'complete',
           'recipe': 'h9_digits',
           'true_comparisons': [{'split_seed': 20261105,
                                 'h5': 0.9443534995440844,
                                 'h9_digits': 0.9472359102190997,
                                 'delta': 0.002882410675015268},
                                {'split_seed': 20261106,
                                 'h5': 0.9421700058833905,
                                 'h9_digits': 0.9459635582990402,
                                 'delta': 0.0037935524156497147}],
           'negative_controls': [{'name': 'negative_20261105_2027',
                                  'split_seed': 20261105,
                                  'label_seed': 2027,
                                  'prediction_sha256_before_scoring': 'af3d21d21e614f351dc359cfd5e77d8296cec728f6e68ee66d8e748a2c35e5a3',
                                  'auc': 0.4900267836339812,
                                  'p': 0.0884,
                                  'extreme': 441,
                                  'draws': 4999,
                                  'mc_seed': 202611051,
                                  'rows': 139992,
                                  'positives': 2397,
                                  'bonferroni_p': 0.5304},
                                 {'name': 'negative_20261105_2137',
                                  'split_seed': 20261105,
                                  'label_seed': 2137,
                                  'prediction_sha256_before_scoring': '2e87eaa08bffee479033c1f06e648ef8dd71e8d574831cae26fb555802c568d5',
                                  'auc': 0.5043375655380173,
                                  'p': 0.4644,
                                  'extreme': 2321,
                                  'draws': 4999,
                                  'mc_seed': 202611052,
                                  'rows': 139992,
                                  'positives': 2359,
                                  'bonferroni_p': 1.0},
                                 {'name': 'negative_20261105_2267',
                                  'split_seed': 20261105,
                                  'label_seed': 2267,
                                  'prediction_sha256_before_scoring': '384c5d8db58e022c54172835778732e7c1d8e6731d0fb735f639f80133aa36ea',
                                  'auc': 0.496530721097783,
                                  'p': 0.563,
                                  'extreme': 2814,
                                  'draws': 4999,
                                  'mc_seed': 202611053,
                                  'rows': 139992,
                                  'positives': 2316,
                                  'bonferroni_p': 1.0},
                                 {'name': 'negative_20261106_2027',
                                  'split_seed': 20261106,
                                  'label_seed': 2027,
                                  'prediction_sha256_before_scoring': '5e1b1e52e984ee8f5ec0d7438e3ca51e54401aff2f54ad6438855f8e5550e36d',
                                  'auc': 0.5014914449945069,
                                  'p': 0.7944,
                                  'extreme': 3971,
                                  'draws': 4999,
                                  'mc_seed': 202611061,
                                  'rows': 139992,
                                  'positives': 2438,
                                  'bonferroni_p': 1.0},
                                 {'name': 'negative_20261106_2137',
                                  'split_seed': 20261106,
                                  'label_seed': 2137,
                                  'prediction_sha256_before_scoring': 'bf33d1732844a5fbfa38fad123b680f6bb30ef8715b6030a7c116227774014ec',
                                  'auc': 0.5015129776908465,
                                  'p': 0.8,
                                  'extreme': 3999,
                                  'draws': 4999,
                                  'mc_seed': 202611062,
                                  'rows': 139992,
                                  'positives': 2413,
                                  'bonferroni_p': 1.0},
                                 {'name': 'negative_20261106_2267',
                                  'split_seed': 20261106,
                                  'label_seed': 2267,
                                  'prediction_sha256_before_scoring': '3e2ceea2d475f90a42feff94ba117b4dab7009aef9e12f897e6cae56b29a6e8a',
                                  'auc': 0.5073890543206229,
                                  'p': 0.2144,
                                  'extreme': 1071,
                                  'draws': 4999,
                                  'mc_seed': 202611063,
                                  'rows': 139992,
                                  'positives': 2362,
                                  'bonferroni_p': 1.0}],
           'canaries': [{'name': 'canary_20261105',
                         'split_seed': 20261105,
                         'prediction_sha256_before_scoring': 'bdd6ae479e14098d7928ec2ba6f579a9b211926bddd7ae861944d33bc28baa22',
                         'auc': 1.0,
                         'p': 0.0002,
                         'extreme': 0,
                         'draws': 4999,
                         'mc_seed': 202611059,
                         'rows': 139992,
                         'positives': 2397,
                         'bonferroni_p': 0.0004},
                        {'name': 'canary_20261106',
                         'split_seed': 20261106,
                         'prediction_sha256_before_scoring': '2a38e5b31b0ed4d075643035a1d8f799e82e729e414cb7c0285fd9db0e8931e8',
                         'auc': 1.0,
                         'p': 0.0002,
                         'extreme': 0,
                         'draws': 4999,
                         'mc_seed': 202611069,
                         'rows': 139992,
                         'positives': 2438,
                         'bonferroni_p': 0.0004}],
           'input_audits': [{'split_seed': 20261105,
                             'ordinary_target_poison_invariant': True,
                             'fit_valid_disjoint': True},
                            {'split_seed': 20261106,
                             'ordinary_target_poison_invariant': True,
                             'fit_valid_disjoint': True}],
           'runner_sha256': '7db01282d41372109e0da87e8abb70f166f5dc3762c492ced219a2a2e3ae2577',
           'criteria': {'both_true_gains': True,
                        'six_negative_checks': True,
                        'two_copied_target_canaries': True,
                        'input_audits': True},
           'numerically_eligible_for_review': True,
           'decision': 'No automatic admission or submission; review the selected fixed CV '
                       'file and all limitations.'},
 'screen_report_sha256': '21591053cbc21f3f9b4281bdfcc4edc213e2556f762bc77d7e4ade968e15a9b3',
 'binding': {'created_before_audit_fits_utc': '2026-11-01T21:19:36.353185+00:00',
             'selected_recipe': 'h9_digits',
             'cv_csv_sha256': '9405c5036ee8f94cf4b6e539beb43a71385c222812ce47b51df906fb719d3b7d',
             'core_sha256': '7db01282d41372109e0da87e8abb70f166f5dc3762c492ced219a2a2e3ae2577',
             'protocol_sha256': '52b8e9b228d8b49a8dd3f90f524b6c2c3a377d7649d925a5b0337568456fa7cb',
             'screen_report_sha256': '21591053cbc21f3f9b4281bdfcc4edc213e2556f762bc77d7e4ade968e15a9b3',
             'device': 'Private Kaggle CPU; same frozen code/parameters/splits/criteria, both '
                       'true-label recipes use the same worker environment.',
             'scope': 'Audit only. No test predictions enter the bound CV CSV; no upload is '
                      'permitted by this worker.'},
 'audit_environment': {'python': '3.13.15',
                       'platform': 'Linux-6.18.55+-x86_64-with-glibc2.39',
                       'libraries': {'lightgbm': '4.6.0',
                                     'numpy': '2.1.3',
                                     'pandas': '2.3.3',
                                     'scikit-learn': '1.6.1',
                                     'scipy': '1.16.3'}},
 'decision': {'status': 'APPROVE_ONE_FIXED_FILE',
              'recorded_before_public_probe_utc': '2026-11-01T21:39:51.276803+00:00',
              'csv_sha256': '9405c5036ee8f94cf4b6e539beb43a71385c222812ce47b51df906fb719d3b7d',
              'selected_recipe': 'h9_digits',
              'current_recipe_chosen_before_public_result': 'h9_digits',
              'maximum_new_submissions': 1,
              'source_file': 'research_20261102/digit_ablation/outputs/h9_digits/submission.csv',
              'reason': 'Registered screen selection and all registered audit criteria '
                        'passed, independently recomputed. Science and reader explicitly '
                        'approved one exploratory original-file probe and H9 default before '
                        'its public result. Root inspected full results, source binding, MC '
                        'tail counts, data/IDs and limitations.',
              'limits': ['Reused/overlapping development data; selected OOF is exploratory.',
                         'Negative non-rejection does not establish absence of weak leakage; '
                         'canaries check direct copied-target access only.',
                         'Generator mechanism and event-time availability remain unresolved.',
                         'No H8/H10 fallback, blend, reseeding, tuning by public outcome or '
                         'second upload.',
                         'Publish this outcome in either direction. Original H5 failure/H6 '
                         'hold/H10 rejection remain.'],
              'reviews': ['checks/stage10/ONE_UPLOAD_SCIENCE_VERDICT.md',
                          'checks/reviews/stage10_one_upload_reader.md'],
              'later_notebook_release_gates_required': True},
 'submission': {'checked_utc': '2026-11-01T21:41:01.777658+00:00',
                'ref': 57884605,
                'status': 'SubmissionStatus.COMPLETE',
                'file_name': 'candidate_submission.csv',
                'description': 'H9 value-digit study 20261102: fixed 5x42 CSV; OOF0.949374; '
                               'registered audit; one probe',
                'submitted_at': '2026-11-01 21:39:55.597000',
                'public_auc': 0.95009,
                'csv_sha256': '9405c5036ee8f94cf4b6e539beb43a71385c222812ce47b51df906fb719d3b7d',
                'submissions_total': 3,
                'new_submissions_under_this_decision': 1},
 'standing': {'snapshot_utc': '2026-11-01 21:41:04 UTC',
              'rank': 41,
              'ranked_teams': 209,
              'unranked_rows_excluded': 1,
              'team_id': 17243922,
              'public_auc': 0.95009,
              'team_name': 'Georgy Mamarin',
              'accepted_submission_count': 3,
              'archive': 'checks/stage10/post_submit_leaderboard/playground-series-s6e11-publicleaderboard-2026-11-01T21:41:04.csv',
              'csv_sha256': 'f47844ed73ecac6d974b6f2ff2f5b6dc2035539377f384eacb517d4198929c04',
              'scope': 'Dated public leaderboard snapshot, not a final rank or comparable '
                       'cohort improvement.'},
 'opening_prose': '# S6E11: First Checks and a Baseline\n'
                  '\n'
                  'By [Georgy Mamarin](https://www.kaggle.com/georgymamarin)\n'
                  '\n'
                  'Adding value and digit encodings to our 600-tree model of the raw '
                  'predictors (H5) raised out-of-fold (OOF) AUC from **0.9464870 to 0.9493735 '
                  '(H9)**, with gains on all five folds. The highest-scoring alternative '
                  'reached OOF 0.9502590, but failed the predeclared shuffled-label filter '
                  'and was not submitted. Sections 15 and 16 show the ablations, controls and '
                  'separate audit behind the choice.\n'
                  '\n'
                  'One fixed H9 file scored **0.95009 on the public leaderboard** and '
                  'improved on H5 by 0.00300. Our team stood **41/209 ranked teams** in the '
                  '2026-11-01 21:41:04 UTC snapshot. Public AUC and local OOF measure '
                  'different rows; this is a dated public standing, not a final rank. The '
                  "notebook's default fit now produces H9 as the primary "
                  '**`submission.csv`**, with its own metrics and manifest. `CURRENT_CFG` '
                  'lets you change its inputs and settings; the original 300-tree reference '
                  'stays in `reference_submission.csv`.\n'
                  '\n'
                  'Use this notebook to audit all 24 supplied features, compare '
                  'representations and export checked predictions for [Playground S6E11: '
                  'Predicting Credit Card '
                  'Fraud](https://www.kaggle.com/competitions/playground-series-s6e11). The '
                  'recorded investigations explain each selection decision, including the '
                  'original H5 control failure. No cell uploads predictions.',
 'decision_prose': '### Decision and public probe\n'
                   '\n'
                   'H9 passed the original full-control filter and all four criteria in the '
                   'separate audit. Its two true-label gains over H5 were **+0.0028824** and '
                   '**+0.0037936**. The smallest adjusted p among the six negative checks was '
                   '**0.5304**. Independent review supported one exploratory upload of the '
                   'five-fold H9 file that had been bound before the audit. This is an '
                   'operational decision under the recorded criteria; negative non-rejection '
                   'and copied-target detection do not establish that the pipeline is free of '
                   'leakage.\n'
                   '\n'
                   "The accepted submission **57884605** scored **0.95009**, versus H5's "
                   "0.94709 and the first reference's 0.94587. It improved on H5 by 0.00300. "
                   'The team ranked **41 of 209 ranked teams** at **2026-11-01 21:41:04 '
                   'UTC**; teams without a ranked score are excluded. We chose H9 as the '
                   'current recipe before this public result. We did not change its '
                   'parameters or seeds, add a blend or switch to another recipe in response. '
                   'The original H5 failure, H6 hold and H10 rejection remain recorded.\n'
                   '\n'
                   'The screen ran locally with LightGBM 4.6.0, scikit-learn 1.6.1, NumPy '
                   '2.2.6 and pandas 2.2.3. The audit ran on a private Kaggle CPU worker with '
                   'Python 3.13.15, NumPy 2.1.3 and pandas 2.3.3; LightGBM and scikit-learn '
                   'versions matched. Both true-label recipes were compared within that '
                   'worker. This was not a bitwise local reproduction. None of the audit '
                   'models supplied the submitted test predictions.\n'
                   '\n'
                   'The OOF comparison reuses development data after several earlier studies. '
                   'Selection can make its winning score optimistic, and the overlapping '
                   'audit partitions do not provide external validation. We have established '
                   'a useful representation gain under these splits. We have not '
                   'reconstructed the synthetic generator or established when every supplied '
                   'feature became available.',
 'hero_caption': 'These are recorded OOF AUCs on the same five folds. Blue marks H9, selected '
                 'for the separate audit; the colour alone does not imply admission. H10 has '
                 'a higher true-label score but failed its full shuffled-label filter. H9 '
                 'versus H8 changes the whole digit bundle, including column-subsampling '
                 'paths.',
 'current_export': {'decision_before_public_probe': True,
                    'settings': {'recipe_key': 'h9_digits',
                                 'view': 'digits',
                                 'model_params': {'n_estimators': 600,
                                                  'num_leaves': 15,
                                                  'learning_rate': 0.03,
                                                  'random_state': 42,
                                                  'n_jobs': 4,
                                                  'min_child_samples': 40,
                                                  'reg_lambda': 12,
                                                  'subsample': 0.85,
                                                  'subsample_freq': 1,
                                                  'colsample_bytree': 0.85,
                                                  'verbosity': -1,
                                                  'deterministic': True,
                                                  'force_col_wise': True},
                                 'drop_predictors': [],
                                 'n_splits': 5,
                                 'split_seed': 42}}}
digit_reproduction_sources = {'run_digit_study.py': '"""Fixed S6E11 value/digit ablations. Imports never fit models or upload files."""\nfrom pathlib import Path\nfrom datetime import datetime, timezone\nimport hashlib, importlib.metadata, json, time, gc\nimport numpy as np\nimport pandas as pd\nfrom sklearn.preprocessing import OrdinalEncoder, TargetEncoder\nfrom sklearn.model_selection import StratifiedKFold, KFold\nfrom sklearn.metrics import roc_auc_score\nfrom scipy.stats import rankdata\nfrom lightgbm import LGBMClassifier\n\nFLOATS = [\'amount_usd\',\'hours_since_last_txn\',\'distance_from_home_km\',\n          \'account_balance_usd\',\'velocity_score\',\'merchant_risk_score\']\nRECIPES = [\'h7_cents\',\'h8_per_value\',\'h9_digits\',\'h10_small_trees\']\n\ndef sha_file(path):\n    return hashlib.sha256(Path(path).read_bytes()).hexdigest()\n\ndef digest(array):\n    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()\n\ndef write_json(path, value):\n    path=Path(path);tmp=path.with_suffix(\'.tmp\')\n    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+\'\\n\');tmp.replace(path)\n\ndef digit_keys(frame):\n    """Round only the derived keys to cents; retain every raw predictor unchanged."""\n    result={}\n    for name in FLOATS:\n        x=frame[name].to_numpy(dtype=np.float64)\n        if not (np.isfinite(x).all() and (x>=0).all()):\n            raise ValueError(\'The recorded digit-key protocol requires finite nonnegative values.\')\n        k=np.rint(100*x).astype(np.int64);integer=k//100\n        length=np.ones(len(k),dtype=np.int64)\n        positive=integer>0\n        length[positive]=np.floor(np.log10(integer[positive])).astype(np.int64)+1\n        result[\'cent_\'+name]=k%100\n        result[\'integer_\'+name]=integer\n        result[\'tens_\'+name]=k//1000\n        result[\'tenths_\'+name]=k//10\n        result[\'leading1_\'+name]=integer//np.power(10,length-1)\n        result[\'leading2_\'+name]=integer//np.power(10,np.maximum(length-2,0))\n        result[\'length_\'+name]=length\n    return pd.DataFrame(result,index=frame.index)\n\ndef input_audit(train,test):\n    rows=[]\n    for label,frame in [(\'train\',train),(\'test\',test)]:\n        for column in FLOATS:\n            scaled=frame[column].to_numpy(dtype=float)*100\n            residual=np.abs(scaled-np.rint(scaled));bad=np.flatnonzero(residual>1e-6)\n            rows.append({\'file\':label,\'column\':column,\'non_cent_aligned_rows\':len(bad),\n                \'max_cent_residual\':float(residual.max()),\n                \'exceptions\':[{\'id\':int(frame.iloc[i].id),\'raw\':float(frame.iloc[i][column])} for i in bad],\n                \'zero_count\':int(frame[column].eq(0).sum())})\n    return rows\n\ndef native_matrices(fitting,others,columns):\n    cats=[c for c in columns if not pd.api.types.is_numeric_dtype(fitting[c])]\n    nums=[c for c in columns if c not in cats]\n    encoder=OrdinalEncoder(handle_unknown=\'use_encoded_value\',unknown_value=-1)\n    if cats:encoder.fit(fitting[cats].astype(\'string\').fillna(\'<missing>\').astype(str))\n    out=[]\n    for frame in [fitting]+others:\n        values=frame[nums].to_numpy(dtype=np.float32)\n        if cats:values=np.column_stack([values,encoder.transform(frame[cats].astype(\'string\').fillna(\'<missing>\').astype(str))]).astype(np.float32)\n        out.append(pd.DataFrame(values,columns=nums+cats,index=frame.index))\n    return out,list(range(len(nums),len(columns)))\n\ndef make_views(fitting,others,labels,features,modes):\n    """Held-out labels are not arguments. TE training rows use inner cross-fitting."""\n    frames=[fitting.loc[:,features].reset_index(drop=True).copy()]+[\n        f.loc[:,features].reset_index(drop=True).copy() for f in others]\n    keys=[digit_keys(f) for f in frames] if any(m in {\'cents\',\'digits\'} for m in modes) else None\n    encoded=[{} for _ in frames];audit=[]\n    if any(m in {\'per_value\',\'digits\'} for m in modes):\n        key_columns=[(\'value_\'+c,[f[c] for f in frames]) for c in features]\n        if \'digits\' in modes:key_columns += [(\'digit_\'+c,[k[c] for k in keys]) for c in keys[0].columns]\n        for name,series in key_columns:\n            arrays=[s.to_numpy().reshape(-1,1) for s in series]\n            encoder=TargetEncoder(target_type=\'binary\',smooth=20.0,cv=5,shuffle=True,random_state=0)\n            encoded[0][\'te_\'+name]=encoder.fit_transform(arrays[0],labels).ravel().astype(np.float32)\n            unseen=[]\n            for j in range(1,len(frames)):\n                encoded[j][\'te_\'+name]=encoder.transform(arrays[j]).ravel().astype(np.float32)\n                unseen.append(int((~series[j].isin(encoder.categories_[0])).sum()))\n            audit.append({\'name\':name,\'fitting_unique\':len(encoder.categories_[0]),\'unseen_other_rows\':unseen})\n    result={}\n    for mode in modes:\n        prepared=[]\n        for j,frame in enumerate(frames):\n            additions={}\n            if mode in {\'cents\',\'digits\'}:\n                additions.update({c:keys[j][c].to_numpy(dtype=np.float32) for c in keys[j] if c.startswith(\'cent_\')})\n            if mode in {\'per_value\',\'digits\'}:\n                additions.update({c:v for c,v in encoded[j].items() if mode==\'digits\' or c.startswith(\'te_value_\')})\n            prepared.append(pd.concat([frame,pd.DataFrame(additions,index=frame.index)],axis=1))\n        result[mode]=native_matrices(prepared[0],prepared[1:],list(prepared[0].columns))\n    return result,audit\n\ndef conditional_reference(labels,prediction,seed):\n    labels=np.asarray(labels);n=len(labels);n1=int(labels.sum());n0=n-n1\n    assert n1>0 and n0>0\n    ranks2=np.rint(2*rankdata(prediction,method=\'average\')).astype(np.int64)\n    constant=n1*(n1+1);centre=n1*n0\n    observed=int(ranks2[labels==1].sum())-constant\n    auc=observed/(2*n1*n0)\n    assert abs(auc-roc_auc_score(labels,prediction))<1e-12\n    statistic=abs(observed-centre);rng=np.random.Generator(np.random.PCG64(seed))\n    null=np.empty(4999);extreme=0\n    for i in range(4999):\n        value=int(ranks2[rng.choice(n,n1,replace=False)].sum())-constant\n        extreme+=abs(value-centre)>=statistic;null[i]=value/(2*n1*n0)\n    return {\'auc\':float(auc),\'p\':(1+int(extreme))/5000,\'extreme\':int(extreme),\n            \'draws\':4999,\'mc_seed\':seed,\'rows\':n,\'positives\':n1},null\n\ndef checked_data(data,protocol):\n    data=Path(data)\n    for name,expected in protocol[\'input_sha256\'].items():assert sha_file(data/name)==expected,name\n    train=pd.read_csv(data/\'train.csv\');test=pd.read_csv(data/\'test.csv\');sample=pd.read_csv(data/\'sample_submission.csv\')\n    features=[c for c in train if c not in {\'id\',\'is_fraud\'}]\n    assert features==protocol[\'features\'] and len(train)==699960 and len(test)==299984\n    assert train.is_fraud.sum()==11689 and train.id.is_unique and test.id.is_unique\n    return train,test,sample,features\n\ndef run_study(data,output,reference_oof,reference_folds,protocol):\n    out=Path(output);out.mkdir(parents=True,exist_ok=True)\n    assert sha_file(reference_oof)==protocol[\'reference\'][\'oof_sha256\']\n    assert sha_file(reference_folds)==protocol[\'reference\'][\'fold_ids_sha256\']\n    train,test,sample,features=checked_data(data,protocol);y=train.is_fraud.to_numpy()\n    reference=np.load(reference_oof);saved_ids=np.load(reference_folds)\n    assert reference.shape==y.shape and saved_ids.shape==y.shape\n    write_json(out/\'recorded_protocol.json\',protocol)\n    write_json(out/\'input_audit.json\',input_audit(train,test))\n    report={\'status\':\'running\',\'started_utc\':datetime.now(timezone.utc).isoformat(),\n        \'runner_sha256\':sha_file(__file__),\'reference_auc\':float(roc_auc_score(y,reference)),\n        \'libraries\':{p:importlib.metadata.version(p) for p in [\'lightgbm\',\'numpy\',\'pandas\',\'scikit-learn\',\'scipy\']},\n        \'recipes\':{name:{\'true_folds\':[],\'null_folds\':[]} for name in RECIPES},\'encoder_audits\':[]}\n    predictions={name:{\'true\':np.full(len(y),np.nan),\'null\':np.full(len(y),np.nan),\'test\':np.zeros(len(test))} for name in RECIPES}\n    univariate_te=np.full((len(y),len(FLOATS)),np.nan,dtype=np.float32)\n    all_ids={}\n    for arm,labels in [(\'true\',y),(\'null\',np.random.RandomState(0).permutation(y))]:\n        ids=np.full(len(y),-1,np.int16);all_ids[arm]=ids\n        for k,(fit,valid) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(train,labels)):\n            ids[valid]=k\n            if arm==\'true\':np.testing.assert_array_equal(saved_ids[valid],np.full(len(valid),k))\n            others=[train.iloc[valid]]+([test] if arm==\'true\' else [])\n            for mode,names in [(\'cents\',[\'h7_cents\']),(\'per_value\',[\'h8_per_value\']),\n                               (\'digits\',[\'h9_digits\',\'h10_small_trees\'])]:\n                prepared,audit=make_views(train.iloc[fit],others,labels[fit],features,{mode})\n                report[\'encoder_audits\'].append({\'arm\':arm,\'fold\':k,\'view\':mode,\'fields\':audit})\n                matrices,cats=prepared[mode];a,b=matrices[:2]\n                if arm==\'true\' and mode==\'per_value\':\n                    univariate_te[valid]=b[[\'te_value_\'+c for c in FLOATS]].to_numpy()\n                for name in names:\n                    spec=protocol[\'recipes\'][name];began=time.perf_counter()\n                    model=LGBMClassifier(**spec[\'params\']).fit(a,labels[fit],categorical_feature=cats)\n                    np.testing.assert_array_equal(model.classes_,[0,1]);assert model.n_estimators_==spec[\'params\'][\'n_estimators\']\n                    values=model.predict_proba(b)[:,1];assert np.isfinite(values).all()\n                    predictions[name][arm][valid]=values\n                    if arm==\'true\':predictions[name][\'test\']+=model.predict_proba(matrices[2])[:,1]/5\n                    row={\'fold\':k,\'auc\':float(roc_auc_score(labels[valid],values)),\n                        \'trees\':int(model.n_estimators_),\'features\':a.shape[1],\'seconds\':time.perf_counter()-began}\n                    assert a.shape[1]==spec[\'feature_count\']\n                    if arm==\'true\':row[\'delta_vs_h5\']=row[\'auc\']-float(roc_auc_score(y[valid],reference[valid]))\n                    report[\'recipes\'][name][arm+\'_folds\'].append(row)\n                    print(arm,name,\'fold\',k,row,flush=True)\n                    del model\n                del prepared,matrices,a,b;gc.collect()\n            write_json(out/\'report.json\',report)\n    assert np.isfinite(univariate_te).all()\n    np.save(out/\'univariate_heldout_te.npy\',univariate_te)\n    report[\'univariate_numeric\']=[{\'column\':c,\'raw_auc\':float(roc_auc_score(y,train[c])),\n        \'heldout_per_value_te_auc\':float(roc_auc_score(y,univariate_te[:,j]))} for j,c in enumerate(FLOATS)]\n    for name,arrays in predictions.items():\n        destination=out/name;destination.mkdir(exist_ok=True)\n        for key in arrays:assert np.isfinite(arrays[key]).all() and ((arrays[key]>=0)&(arrays[key]<=1)).all()\n        np.save(destination/\'oof.npy\',arrays[\'true\']);np.save(destination/\'test.npy\',arrays[\'test\']);np.save(destination/\'fold_ids.npy\',all_ids[\'true\'])\n        np.savez_compressed(destination/\'control.npz\',oof=arrays[\'null\'],labels=np.random.RandomState(0).permutation(y),fold_ids=all_ids[\'null\'])\n        auc=float(roc_auc_score(y,arrays[\'true\']));null_auc=float(roc_auc_score(np.random.RandomState(0).permutation(y),arrays[\'null\']))\n        row=report[\'recipes\'][name];deltas=[f[\'delta_vs_h5\'] for f in row[\'true_folds\']]\n        row.update(auc=auc,delta_vs_h5=auc-report[\'reference_auc\'],control_auc=null_auc,\n            positive_folds=sum(d>0 for d in deltas),fold_deltas=deltas,\n            old_null_pass=bool(abs(null_auc-.5)<.003),\n            eligible_for_final_audit=bool(auc-report[\'reference_auc\']>=.0005 and sum(d>0 for d in deltas)>=4 and abs(null_auc-.5)<.003))\n        aligned=sample.copy();order=pd.Index(test.id).get_indexer(aligned.id);assert (order>=0).all()\n        aligned[\'is_fraud\']=arrays[\'test\'][order];aligned.to_csv(destination/\'submission.csv\',index=False)\n        row[\'submission_sha256\']=sha_file(destination/\'submission.csv\');write_json(destination/\'report.json\',row)\n    eligible=[name for name in RECIPES if report[\'recipes\'][name][\'eligible_for_final_audit\']]\n    selected=max(eligible,key=lambda name:report[\'recipes\'][name][\'auc\']) if eligible else None\n    report.update(status=\'complete\',selected_for_final_audit=selected,\n        submission_decision=\'No upload: final audit, independent review and a separate root decision are still required.\',\n        completed_utc=datetime.now(timezone.utc).isoformat())\n    write_json(out/\'report.json\',report);print(json.dumps({name:{k:r[k] for k in [\'auc\',\'delta_vs_h5\',\'control_auc\',\'eligible_for_final_audit\']} for name,r in report[\'recipes\'].items()},indent=2),flush=True)\n    return report\n\ndef run_final_audit(data,output,recipe,protocol):\n    out=Path(output);out.mkdir(parents=True,exist_ok=True)\n    train,test,sample,features=checked_data(data,protocol);y=train.is_fraud.to_numpy()\n    spec=protocol[\'recipes\'][recipe];rules=protocol[\'final_audit\']\n    report={\'status\':\'running\',\'recipe\':recipe,\'true_comparisons\':[],\'negative_controls\':[],\n            \'canaries\':[],\'input_audits\':[],\'runner_sha256\':sha_file(__file__)}\n    def fit_predict(labels,fit,valid,name,candidate=True,canary=False):\n        mode=spec[\'view\'] if candidate else \'raw\'\n        prepared,_=make_views(train.iloc[fit],[train.iloc[valid]],labels[fit],features,{mode})\n        (a,b),cats=prepared[mode]\n        if not canary:\n            poison=train.iloc[valid].copy();poison[\'is_fraud\']=1-poison.is_fraud\n            rebuilt,_=make_views(train.iloc[fit],[poison],labels[fit],features,{mode})\n            (aa,bb),cc=rebuilt[mode];np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb);assert cats==cc\n        if canary:\n            a=a.copy();b=b.copy();a[\'copied_assigned_target\']=labels[fit];b[\'copied_assigned_target\']=labels[valid]\n        params=spec[\'params\'] if candidate else protocol[\'h5_params\']\n        model=LGBMClassifier(**params).fit(a,labels[fit],categorical_feature=cats)\n        assert model.n_estimators_==params[\'n_estimators\']\n        prediction=model.predict_proba(b)[:,1];assert np.isfinite(prediction).all()\n        np.save(out/(name+\'_prediction.npy\'),prediction);sealed=digest(prediction)\n        held=labels[valid]\n        np.savez_compressed(out/(name+\'.npz\'),prediction=prediction,labels=held,fitting_indices=fit,validation_indices=valid)\n        return prediction,held,sealed\n    for split_seed in rules[\'splits\']:\n        fit,valid=next(KFold(5,shuffle=True,random_state=split_seed).split(train));scores={}\n        for candidate,name in [(False,\'h5\'),(True,recipe)]:\n            prediction,held,sealed=fit_predict(y,fit,valid,f\'true_{name}_{split_seed}\',candidate=candidate)\n            scores[name]=float(roc_auc_score(held,prediction))\n        delta=scores[recipe]-scores[\'h5\'];report[\'true_comparisons\'].append({\'split_seed\':split_seed,**scores,\'delta\':delta})\n        report[\'input_audits\'].append({\'split_seed\':split_seed,\'ordinary_target_poison_invariant\':True,\'fit_valid_disjoint\':not len(np.intersect1d(fit,valid))})\n        print(\'true comparison\',report[\'true_comparisons\'][-1],flush=True)\n        for label_seed in rules[\'label_seeds\']:\n            labels=np.random.RandomState(label_seed).permutation(y);name=f\'negative_{split_seed}_{label_seed}\'\n            prediction,held,sealed=fit_predict(labels,fit,valid,name)\n            measured,reference=conditional_reference(held,prediction,rules[\'mc_seeds\'][f\'{split_seed}_{label_seed}\'])\n            np.save(out/(name+\'_reference.npy\'),reference)\n            report[\'negative_controls\'].append({\'name\':name,\'split_seed\':split_seed,\'label_seed\':label_seed,\'prediction_sha256_before_scoring\':sealed,**measured})\n            print(name,measured,flush=True);write_json(out/\'report.json\',report)\n        labels=np.random.RandomState(rules[\'label_seeds\'][0]).permutation(y);name=f\'canary_{split_seed}\'\n        prediction,held,sealed=fit_predict(labels,fit,valid,name,canary=True)\n        measured,reference=conditional_reference(held,prediction,rules[\'canary_mc_seeds\'][str(split_seed)])\n        np.save(out/(name+\'_reference.npy\'),reference)\n        report[\'canaries\'].append({\'name\':name,\'split_seed\':split_seed,\'prediction_sha256_before_scoring\':sealed,**measured})\n        print(name,measured,flush=True);write_json(out/\'report.json\',report)\n    for row in report[\'negative_controls\']:row[\'bonferroni_p\']=min(1.0,6*row[\'p\'])\n    for row in report[\'canaries\']:row[\'bonferroni_p\']=min(1.0,2*row[\'p\'])\n    criteria={\'both_true_gains\':all(r[\'delta\']>=.0005 for r in report[\'true_comparisons\']),\n        \'six_negative_checks\':all(r[\'bonferroni_p\']>=.05 for r in report[\'negative_controls\']),\n        \'two_copied_target_canaries\':all(r[\'bonferroni_p\']<.01 for r in report[\'canaries\']),\n        \'input_audits\':all(all(v for k,v in r.items() if k!=\'split_seed\') for r in report[\'input_audits\'])}\n    report.update(status=\'complete\',criteria=criteria,numerically_eligible_for_review=all(criteria.values()),\n        decision=\'No automatic admission or submission; review the selected fixed CV file and all limitations.\')\n    write_json(out/\'report.json\',report);return report\n', 'reproduce_digit_study.py': '"""Opt-in method replay. Keep this file beside run_digit_study.py in notebook outputs."""\nfrom pathlib import Path\nimport importlib.util,json,copy,hashlib\nimport numpy as np,pandas as pd\nfrom sklearn.model_selection import StratifiedKFold\nfrom sklearn.metrics import roc_auc_score\nfrom lightgbm import LGBMClassifier\n\ndef _load_core():\n    source=Path(__file__).with_name(\'run_digit_study.py\')\n    if not source.exists():\n        raise FileNotFoundError(\'Download run_digit_study.py beside this replay helper.\')\n    spec=importlib.util.spec_from_file_location(\'fixed_digit_study_core\',source)\n    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n    return module\n\ndef run_screen(data,output,recorded_protocol):\n    """Five fresh H5 reference fits plus the original forty-fit four-recipe screen."""\n    core=_load_core();out=Path(output);out.mkdir(parents=True,exist_ok=True)\n    train,test,sample,features=core.checked_data(data,recorded_protocol)\n    y=train.is_fraud.to_numpy();oof=np.full(len(train),np.nan)\n    test_prediction=np.zeros(len(test));ids=np.full(len(train),-1,np.int16)\n    for k,(fit,valid) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(train,y)):\n        views,_=core.make_views(train.iloc[fit],[train.iloc[valid],test],y[fit],features,{\'raw\'})\n        (a,b,t),cats=views[\'raw\']\n        model=LGBMClassifier(**recorded_protocol[\'h5_params\']).fit(a,y[fit],categorical_feature=cats)\n        oof[valid]=model.predict_proba(b)[:,1];test_prediction+=model.predict_proba(t)[:,1]/5;ids[valid]=k\n        print(\'Recomputed H5 reference fold\',k,roc_auc_score(y[valid],oof[valid]),flush=True)\n        del views,a,b,t,model\n    assert np.isfinite(oof).all() and np.isfinite(test_prediction).all()\n    np.save(out/\'reference_h5_oof.npy\',oof);np.save(out/\'reference_h5_test.npy\',test_prediction)\n    np.save(out/\'reference_h5_fold_ids.npy\',ids)\n    replay=copy.deepcopy(recorded_protocol)\n    replay[\'reference\'].update(name=\'Fresh fixed H5 reference\',auc=float(roc_auc_score(y,oof)),\n        oof_sha256=core.sha_file(out/\'reference_h5_oof.npy\'),\n        fold_ids_sha256=core.sha_file(out/\'reference_h5_fold_ids.npy\'))\n    replay[\'replay_scope\']=\'Fresh reference plus recorded four-recipe method; historical pre-fit file binding is not recreated. No automatic admission/upload.\'\n    replay[\'replay_reference_fits\']=5\n    report=core.run_study(data,out/\'screen\',out/\'reference_h5_oof.npy\',out/\'reference_h5_fold_ids.npy\',replay)\n    report.update(historical_file_binding_recreated=False,\n        submission_decision=\'Numeric method replay only. No original admission or new upload is recreated.\',\n        replay_total_model_fits=45)\n    core.write_json(out/\'screen/report.json\',report)\n    return report\n\ndef run_audit(data,output,recipe,recorded_protocol):\n    """Replay twelve audit fits; numeric criteria do not recreate the historical decision."""\n    core=_load_core()\n    report=core.run_final_audit(data,output,recipe,recorded_protocol)\n    report.update(historical_file_binding_recreated=False,admission=False,\n        decision=\'Numeric method replay only; the historical pre-fit selected-file binding and independent submission review are not recreated.\')\n    core.write_json(Path(output)/\'report.json\',report)\n    return report\n'}
for s10_filename,s10_source in digit_reproduction_sources.items():
    (output/s10_filename).write_text(s10_source)

# %% cell 81
from sklearn.preprocessing import TargetEncoder

FLOATS = ['amount_usd','hours_since_last_txn','distance_from_home_km',
          'account_balance_usd','velocity_score','merchant_risk_score']

RECIPES = ['h7_cents','h8_per_value','h9_digits','h10_small_trees']

def digit_keys(frame):
    """Round only the derived keys to cents; retain every raw predictor unchanged."""
    result={}
    for name in FLOATS:
        x=frame[name].to_numpy(dtype=np.float64)
        if not (np.isfinite(x).all() and (x>=0).all()):
            raise ValueError('The recorded digit-key protocol requires finite nonnegative values.')
        k=np.rint(100*x).astype(np.int64);integer=k//100
        length=np.ones(len(k),dtype=np.int64)
        positive=integer>0
        length[positive]=np.floor(np.log10(integer[positive])).astype(np.int64)+1
        result['cent_'+name]=k%100
        result['integer_'+name]=integer
        result['tens_'+name]=k//1000
        result['tenths_'+name]=k//10
        result['leading1_'+name]=integer//np.power(10,length-1)
        result['leading2_'+name]=integer//np.power(10,np.maximum(length-2,0))
        result['length_'+name]=length
    return pd.DataFrame(result,index=frame.index)

def native_matrices(fitting,others,columns):
    cats=[c for c in columns if not pd.api.types.is_numeric_dtype(fitting[c])]
    nums=[c for c in columns if c not in cats]
    encoder=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    if cats:encoder.fit(fitting[cats].astype('string').fillna('<missing>').astype(str))
    out=[]
    for frame in [fitting]+others:
        values=frame[nums].to_numpy(dtype=np.float32)
        if cats:values=np.column_stack([values,encoder.transform(frame[cats].astype('string').fillna('<missing>').astype(str))]).astype(np.float32)
        out.append(pd.DataFrame(values,columns=nums+cats,index=frame.index))
    return out,list(range(len(nums),len(columns)))

def make_views(fitting,others,labels,features,modes):
    """Held-out labels are not arguments. TE training rows use inner cross-fitting."""
    frames=[fitting.loc[:,features].reset_index(drop=True).copy()]+[
        f.loc[:,features].reset_index(drop=True).copy() for f in others]
    keys=[digit_keys(f) for f in frames] if any(m in {'cents','digits'} for m in modes) else None
    encoded=[{} for _ in frames];audit=[]
    if any(m in {'per_value','digits'} for m in modes):
        key_columns=[('value_'+c,[f[c] for f in frames]) for c in features]
        if 'digits' in modes:key_columns += [('digit_'+c,[k[c] for k in keys]) for c in keys[0].columns]
        for name,series in key_columns:
            arrays=[s.to_numpy().reshape(-1,1) for s in series]
            encoder=TargetEncoder(target_type='binary',smooth=20.0,cv=5,shuffle=True,random_state=0)
            encoded[0]['te_'+name]=encoder.fit_transform(arrays[0],labels).ravel().astype(np.float32)
            unseen=[]
            for j in range(1,len(frames)):
                encoded[j]['te_'+name]=encoder.transform(arrays[j]).ravel().astype(np.float32)
                unseen.append(int((~series[j].isin(encoder.categories_[0])).sum()))
            audit.append({'name':name,'fitting_unique':len(encoder.categories_[0]),'unseen_other_rows':unseen})
    result={}
    for mode in modes:
        prepared=[]
        for j,frame in enumerate(frames):
            additions={}
            if mode in {'cents','digits'}:
                additions.update({c:keys[j][c].to_numpy(dtype=np.float32) for c in keys[j] if c.startswith('cent_')})
            if mode in {'per_value','digits'}:
                additions.update({c:v for c,v in encoded[j].items() if mode=='digits' or c.startswith('te_value_')})
            prepared.append(pd.concat([frame,pd.DataFrame(additions,index=frame.index)],axis=1))
        result[mode]=native_matrices(prepared[0],prepared[1:],list(prepared[0].columns))
    return result,audit

# %% cell 83
s10_audit=pd.DataFrame(recorded_digit_study['input_audit'])
display(styled_table(s10_audit[['file','column','non_cent_aligned_rows']]
    .set_index(['file','column']).rename(columns={'non_cent_aligned_rows':'Rows beyond cent precision'})))
s10_examples=[1371.37,9.982,0.0]
s10_example_frame=pd.DataFrame({c:s10_examples for c in FLOATS})
s10_example_keys=digit_keys(s10_example_frame)
s10_example_table=pd.DataFrame({str(v):{'Raw value':str(v),**{
    name:str(int(s10_example_keys.iloc[j][name+'_amount_usd']))
    for name in ['cent','integer','tens','tenths','leading1','leading2','length']}}
    for j,v in enumerate(s10_examples)})
display(styled_table(s10_example_table))

# %% cell 85
def digit_comparison_figure(study):
    s10_names=['H5: raw comparator','H7: cent keys','H8: value encoding','H9: value + digit keys','H10: four-leaf limit']
    s10_values=[study['screen']['reference_auc']]+[study['screen']['recipes'][k]['auc'] for k in RECIPES]
    s10_selected=study['screen']['selected_for_final_audit']
    fig,ax=plt.subplots(figsize=(7.3,5.0))
    for j,value in enumerate(s10_values):
        colour='#0072B2' if j>0 and RECIPES[j-1]==s10_selected else '#52616E'
        ax.scatter(value,j,color=colour,s=65,zorder=3)
        ax.annotate(f'{value:.7f}',(value,j),xytext=(8,0),textcoords='offset points',va='center',fontsize=14)
    ax.set_yticks(range(5),s10_names);ax.set_ylim(4.6,-.6)
    span=max(s10_values)-min(s10_values)
    ax.set_xlim(min(s10_values)-max(.0005,span*.10),max(s10_values)+max(.002,span*.55))
    ax.set(title='Value encodings on the frozen folds',xlabel='Recorded true-label OOF AUC')
    ax.xaxis.set_major_locator(plt.MaxNLocator(4));ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x:.4f}'))
    ax.grid(axis='x');fig.subplots_adjust(left=.37,right=.97,top=.84,bottom=.20)
    return fig


s10_fig=digit_comparison_figure(recorded_digit_study)
s10_fig.savefig('digit_ablation_snapshot.png',dpi=100,facecolor='white');plt.close(s10_fig)

# %% cell 86
s10_rows=[]
for s10_key,s10_result in recorded_digit_study['screen']['recipes'].items():
    s10_rows.append({'Recipe':s10_key.split('_')[0].upper(),
        'Inputs':recorded_digit_study['protocol']['recipes'][s10_key]['feature_count'],
        'OOF AUC':s10_result['auc'],'Delta vs H5':s10_result['delta_vs_h5'],
        'Better folds':f"{s10_result['positive_folds']}/5"})
display(styled_table(pd.DataFrame(s10_rows).set_index('Recipe'),
    {'OOF AUC':'{:.7f}','Delta vs H5':'{:+.7f}'}))
for s10_keys in [('h7_cents','h8_per_value'),('h9_digits','h10_small_trees')]:
    s10_fold_table=pd.DataFrame({k.split('_')[0].upper():recorded_digit_study['screen']['recipes'][k]['fold_deltas'] for k in s10_keys})
    display(styled_table(s10_fold_table.rename_axis('fold'),{c:'{:+.7f}' for c in s10_fold_table.columns}))
s10_parameter_table=pd.DataFrame({
    'H5/H7/H8/H9':{k:str(recorded_digit_study['protocol']['h5_params'][k]) for k in
        ['n_estimators','num_leaves','learning_rate','min_child_samples','reg_lambda','subsample','colsample_bytree']},
    'H10':{k:str(recorded_digit_study['protocol']['recipes']['h10_small_trees']['params'][k]) for k in
        ['n_estimators','num_leaves','learning_rate','min_child_samples','reg_lambda','subsample','colsample_bytree']}})
display(styled_table(s10_parameter_table))

# %% cell 88
s10_uni=pd.DataFrame(recorded_digit_study['screen']['univariate_numeric'])
s10_labels=['Amount','Activity gap (hours)','Distance','Balance','Velocity score','Merchant risk score']
s10_fig,s10_ax=plt.subplots(figsize=(7.3,5.7))
for j,row in s10_uni.iterrows():
    s10_ax.plot([row.raw_auc,row.heldout_per_value_te_auc],[j,j],color='#C5CDD3',linewidth=1,zorder=1)
s10_ax.scatter(s10_uni.raw_auc,range(6),s=55,color='#52616E',label='Raw magnitude',zorder=3)
s10_ax.scatter(s10_uni.heldout_per_value_te_auc,range(6),s=55,marker='D',color='#0072B2',label='Held-out value encoding',zorder=3)
s10_ax.axvline(.5,color='#B8C4CC',linestyle='--',linewidth=1)
s10_ax.set_yticks(range(6),s10_labels);s10_ax.set_ylim(5.6,-.6)
s10_ax.set(title='Value identity changes the ranking',xlabel='Single-feature AUC')
s10_ax.set_xlim(min(.48,s10_uni.raw_auc.min()-.02),max(s10_uni.heldout_per_value_te_auc.max(),s10_uni.raw_auc.max())+.025)
s10_ax.xaxis.set_major_locator(plt.MaxNLocator(4));s10_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x:.2f}'))
s10_ax.grid(axis='x');s10_ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),fontsize=14,ncol=1)
s10_fig.subplots_adjust(left=.33,right=.97,top=.85,bottom=.28)
show_figure(s10_fig,
    'The six float columns were specified before fitting. A raw score ranks by increasing magnitude; '
    'we did not choose its direction from the labels. The encoded score for each row uses only its outer fitting folds. '
    'These exploratory single-feature scores are not conditional feature importance or proof of a generator mechanism.',
    'value_identity_auc.png')

# %% cell 91
s10_null_rows=[]
for s10_key,s10_result in recorded_digit_study['screen']['recipes'].items():
    s10_null_rows.append({'Recipe':s10_key.split('_')[0].upper(),
        'Shuffled AUC':s10_result['control_auc'],'Distance from 0.5':abs(s10_result['control_auc']-.5),
        'Old filter': 'Pass' if s10_result['old_null_pass'] else 'Hold',
        'Eligible for audit':'Yes' if s10_result['eligible_for_final_audit'] else 'No'})
display(styled_table(pd.DataFrame(s10_null_rows).set_index('Recipe'),
    {'Shuffled AUC':'{:.7f}','Distance from 0.5':'{:.7f}'}))
print('Fixed selection for the separate audit:',recorded_digit_study['screen']['selected_for_final_audit'])
print('These are recorded controls; passing an operational filter is not a leakage or feature-availability certificate.')

# %% cell 93
s10_final=recorded_digit_study.get('audit')
if s10_final:
    s10_selected=s10_final['recipe']
    s10_true=pd.DataFrame(s10_final['true_comparisons'])
    s10_true['Partition']=['A','B']
    display(styled_table(s10_true[['Partition','h5',s10_selected,'delta']].set_index('Partition')
        .rename(columns={'h5':'H5 AUC',s10_selected:'Selected AUC','delta':'Delta'}),
        {'H5 AUC':'{:.7f}','Selected AUC':'{:.7f}','Delta':'{:+.7f}'}))
    for s10_seed,s10_label in zip(recorded_digit_study['protocol']['final_audit']['splits'],['A','B']):
        display(callout(f'Partition {s10_label}: these three negative controls belong to the single six-test family.'))
        s10_frame=pd.DataFrame([r for r in s10_final['negative_controls'] if r['split_seed']==s10_seed])
        display(styled_table(s10_frame[['label_seed','auc','p','bonferroni_p']].set_index('label_seed')
            .rename(columns={'auc':'AUC','p':'Raw MC p','bonferroni_p':'Bonferroni p (6)'}),
            {'AUC':'{:.7f}','Raw MC p':'{:.4f}','Bonferroni p (6)':'{:.4f}'}))
    s10_canary=pd.DataFrame(s10_final['canaries']);s10_canary['Partition']=['A','B']
    display(styled_table(s10_canary[['Partition','auc','p','bonferroni_p']].set_index('Partition')
        .rename(columns={'auc':'Copied-target AUC','p':'Raw MC p','bonferroni_p':'Bonferroni p (2)'}),
        {'Copied-target AUC':'{:.7f}','Raw MC p':'{:.4f}','Bonferroni p (2)':'{:.4f}'}))
    display(styled_table(pd.DataFrame({'Passed':s10_final['criteria']})))
else:
    print('No recipe qualified for the predeclared audit. No audit fits or new submission were made.')
(output/'recorded_digit_study.json').write_text(json.dumps(recorded_digit_study,indent=2));

# %% cell 96
"""Live current-recipe fit. Historical reference artifacts retain their meaning."""
import shutil
from datetime import datetime, timezone

def fit_current_recipe():
    cur_primary=output/'submission.csv'
    cur_reference=output/'reference_submission.csv'
    if not cur_reference.exists():
        raise FileNotFoundError('Run the reference export in section 9 first.')
    # Reset the primary export before fitting, so an interrupted/disabled rerun has a labelled reference fallback.
    shutil.copyfile(cur_reference,cur_primary)
    cur_stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    for cur_name in ['current_oof.npy','current_test.npy','current_cv_test.npy','current_folds.csv','current_report.json']:
        cur_path=output/cur_name
        if cur_path.exists():
            cur_archive=output/'previous_current'/cur_stamp
            cur_archive.mkdir(parents=True,exist_ok=True)
            cur_path.rename(cur_archive/cur_name)
    cur_manifest={'primary_csv':'submission.csv','primary_model':'reference',
        'prediction_files':['oof.npy','test.npy'],'metrics_file':'metrics.json',
        'csv_sha256':hashlib.sha256(cur_primary.read_bytes()).hexdigest(),
        'status':'current_training_disabled' if not CFG.get('run_current_recipe',True) else 'current_fit_running_reference_fallback'}
    (output/'submission_manifest.json').write_text(json.dumps(cur_manifest,indent=2))
    if not CFG.get('run_current_recipe',True):
        cur_report={'status':'disabled','primary_model':'reference',
            'message':'The primary submission.csv was restored from reference_submission.csv. No current recipe was fitted.'}
        (output/'current_report.json').write_text(json.dumps(cur_report,indent=2))
        print(cur_report['message']);return cur_report
    cur_settings=CURRENT_CFG
    cur_export_mode=cur_settings.get('export_mode','cv_average')
    if cur_export_mode not in {'cv_average','full_refit'}:
        raise ValueError('Choose cv_average or full_refit as CURRENT_CFG export_mode.')
    cur_features=recorded_digit_study['protocol']['features']
    cur_drop=set(cur_settings['drop_predictors'])
    if cur_drop-set(cur_features):raise ValueError('CURRENT_CFG contains an unknown raw predictor to drop.')
    if len(cur_drop)==len(cur_features):raise ValueError('Keep at least one raw predictor.')
    if cur_settings['view'] not in {'raw','cents','per_value','digits'}:
        raise ValueError('Choose raw, cents, per_value or digits in CURRENT_CFG.')
    cur_y=train['is_fraud'].to_numpy()
    cur_folds=list(StratifiedKFold(cur_settings['n_splits'],shuffle=True,
                                  random_state=cur_settings['split_seed']).split(train,cur_y))
    cur_ids=np.full(len(train),-1,np.int16)
    for cur_k,(_,cur_valid) in enumerate(cur_folds):cur_ids[cur_valid]=cur_k
    cur_oof=np.full(len(train),np.nan);cur_test=np.zeros(len(test));cur_scores=[]
    cur_sources={c:c for c in cur_features}
    cur_sources.update({'te_value_'+c:c for c in cur_features})
    for cur_column in FLOATS:
        cur_sources['cent_'+cur_column]=cur_column
        for cur_key in ['cent','integer','tens','tenths','leading1','leading2','length']:
            cur_sources['te_digit_'+cur_key+'_'+cur_column]=cur_column
    cur_reference_metric_path=output/'reference_metrics.json'
    if not cur_reference_metric_path.exists():cur_reference_metric_path=output/'metrics.json'
    cur_reference_metrics=json.loads(cur_reference_metric_path.read_text())
    cur_reference_config=cur_reference_metrics['config']
    cur_reference_array=np.load(output/'oof.npy')
    cur_reference_oof=cur_reference_array[:,1] if cur_reference_array.ndim==2 else cur_reference_array
    cur_saved_folds=pd.read_csv(output/'folds.csv')
    np.testing.assert_array_equal(cur_saved_folds.id,train.id)
    if cur_reference_oof.shape!=cur_y.shape:raise ValueError('The saved reference predictions do not match these rows.')
    cur_data_fp=hashlib.sha256(pd.util.hash_pandas_object(train[['id','is_fraud']],index=False).values.tobytes()).hexdigest()
    cur_reference_comparable=(cur_reference_config['target']=='is_fraud' and cur_reference_config['positive_class']==1
        and cur_reference_config['task']=='binary' and cur_reference_config['metric']=='auc'
        and cur_reference_metrics['data_fp']==cur_data_fp)
    cur_same_folds=np.array_equal(cur_ids,cur_saved_folds.fold.to_numpy())
    cur_paired=cur_reference_comparable and cur_same_folds
    cur_started=time.perf_counter()
    print('Current recipe key:',cur_settings['recipe_key'])
    print('Current live settings:',json.dumps(cur_settings,indent=2))
    print('CURRENT_CFG sets this fit. The reference comparison uses its saved metrics and folds.')
    cur_report={'status':'running','current_settings':cur_settings,'folds':[],
        'reference_same_folds':bool(cur_same_folds),'reference_target_data_comparable':bool(cur_reference_comparable),
        'paired_reference_comparison':bool(cur_paired),'positive_class':1,'target':'is_fraud',
        'reference_saved_config':cur_reference_config,
        'dropped_raw_predictors':sorted(cur_drop),
        'scope':'Fresh fit from the visible current-recipe code; recorded study results are a separate snapshot.'}
    (output/'current_report.json').write_text(json.dumps(cur_report,indent=2))
    for cur_k,(cur_fit,cur_valid) in enumerate(cur_folds):
        cur_views,_=make_views(train.iloc[cur_fit],[train.iloc[cur_valid],test],
                               cur_y[cur_fit],cur_features,{cur_settings['view']})
        (cur_a,cur_b,cur_t),cur_cats=cur_views[cur_settings['view']]
        cur_cat_names=[cur_a.columns[i] for i in cur_cats]
        cur_keep=[c for c in cur_a if cur_sources[c] not in cur_drop]
        cur_a,cur_b,cur_t=[d.loc[:,cur_keep] for d in [cur_a,cur_b,cur_t]]
        cur_cats=[cur_keep.index(c) for c in cur_cat_names if c in cur_keep]
        cur_model=LGBMClassifier(**cur_settings['model_params']).fit(cur_a,cur_y[cur_fit],categorical_feature=cur_cats)
        np.testing.assert_array_equal(cur_model.classes_,[0,1])
        cur_oof[cur_valid]=cur_model.predict_proba(cur_b)[:,1]
        cur_test+=cur_model.predict_proba(cur_t)[:,1]/len(cur_folds)
        cur_auc=float(roc_auc_score(cur_y[cur_valid],cur_oof[cur_valid]));cur_scores.append(cur_auc)
        cur_row={'fold':cur_k,'auc':cur_auc,'features':len(cur_keep),'actual_trees':int(cur_model.n_estimators_)}
        if cur_paired:cur_row['delta_vs_reference']=cur_auc-float(roc_auc_score(cur_y[cur_valid],cur_reference_oof[cur_valid]))
        cur_report['folds'].append(cur_row)
        (output/'current_report.json').write_text(json.dumps(cur_report,indent=2))
        print('Current fold:',cur_row)
        del cur_views,cur_a,cur_b,cur_t,cur_model
    assert np.isfinite(cur_oof).all() and np.isfinite(cur_test).all()
    assert ((cur_oof>=0)&(cur_oof<=1)).all() and ((cur_test>=0)&(cur_test<=1)).all()
    cur_refit_report=None
    if cur_export_mode=='full_refit':
        # Preserve the five-fold deployment vector separately from the full-training export.
        np.save(output/'current_cv_test.npy',cur_test)
        cur_full_views,_=make_views(train,[test],cur_y,cur_features,{cur_settings['view']})
        (cur_full_a,cur_full_t),cur_full_cats=cur_full_views[cur_settings['view']]
        cur_full_cat_names=[cur_full_a.columns[i] for i in cur_full_cats]
        cur_full_a,cur_full_t=[d.loc[:,cur_keep] for d in [cur_full_a,cur_full_t]]
        cur_full_cats=[cur_keep.index(c) for c in cur_full_cat_names if c in cur_keep]
        cur_full_model=LGBMClassifier(**cur_settings['model_params']).fit(
            cur_full_a,cur_y,categorical_feature=cur_full_cats)
        np.testing.assert_array_equal(cur_full_model.classes_,[0,1])
        assert cur_full_model.n_estimators_==cur_settings['model_params']['n_estimators']
        cur_test=cur_full_model.predict_proba(cur_full_t)[:,1]
        assert np.isfinite(cur_test).all() and ((cur_test>=0)&(cur_test<=1)).all()
        cur_refit_report={'fit_rows':len(train),'features':len(cur_keep),
            'actual_trees':int(cur_full_model.n_estimators_),
            'cv_test_sha256':hashlib.sha256((output/'current_cv_test.npy').read_bytes()).hexdigest(),
            'evaluation':'No OOF estimate for the full-training refit. The reported OOF evaluates its CV recipe.'}
        del cur_full_views,cur_full_a,cur_full_t,cur_full_model
    cur_order=pd.Index(test.id).get_indexer(sample.id)
    assert (cur_order>=0).all() and len(np.unique(cur_order))==len(test)
    cur_submission=sample.copy();cur_submission['is_fraud']=cur_test[cur_order]
    np.testing.assert_array_equal(cur_submission.id,sample.id)
    cur_auc=float(roc_auc_score(cur_y,cur_oof))
    np.save(output/'current_oof.npy',cur_oof);np.save(output/'current_test.npy',cur_test)
    pd.DataFrame({'id':train.id,'fold':cur_ids}).to_csv(output/'current_folds.csv',index=False)
    cur_report.update(status='complete',auc=cur_auc,cv_recipe_auc=cur_auc,
        export_mode=cur_export_mode,full_refit=cur_refit_report,
        predictive_fits=len(cur_folds)+(cur_export_mode=='full_refit'),feature_names=cur_keep,
        reference_auc=float(roc_auc_score(cur_y,cur_reference_oof)) if cur_reference_comparable else None,
        delta_vs_reference=cur_auc-float(roc_auc_score(cur_y,cur_reference_oof)) if cur_reference_comparable else None,
        elapsed_seconds=time.perf_counter()-cur_started,
        data_fingerprint=cur_data_fp,
        split_fingerprint=hashlib.sha256(cur_data_fp.encode()+cur_ids.tobytes()).hexdigest(),
        libraries={p:importlib.metadata.version(p) for p in ['lightgbm','numpy','pandas','scikit-learn']},
        output_oof_sha256=hashlib.sha256((output/'current_oof.npy').read_bytes()).hexdigest(),
        output_test_sha256=hashlib.sha256((output/'current_test.npy').read_bytes()).hexdigest())
    (output/'current_report.json').write_text(json.dumps(cur_report,indent=2))
    cur_temp=output/'current_submission.tmp.csv';cur_submission.to_csv(cur_temp,index=False);cur_temp.replace(cur_primary)
    cur_manifest.update(status='complete',primary_model='current',recipe_key=cur_settings['recipe_key'],
        export_mode=cur_export_mode,oof_scope='CV recipe only; not an evaluation of a full refit',
        prediction_files=['current_oof.npy','current_test.npy'],metrics_file='current_report.json',
        csv_sha256=hashlib.sha256(cur_primary.read_bytes()).hexdigest())
    if cur_export_mode=='full_refit':
        cur_manifest['cv_average_test_file']='current_cv_test.npy'
        cur_manifest['cv_average_test_sha256']=cur_refit_report['cv_test_sha256']
    (output/'submission_manifest.json').write_text(json.dumps(cur_manifest,indent=2))
    cur_formats={'auc':'{:.7f}'}
    if cur_paired:cur_formats['delta_vs_reference']='{:+.7f}'
    display(styled_table(pd.DataFrame(cur_report['folds']).set_index('fold'),cur_formats))
    display(callout(f'The CV recipe has pooled OOF AUC {cur_auc:.7f}. '
                    f'The primary submission.csv uses {cur_export_mode}. '
                    'A full-training refit has no separate OOF estimate. '
                    'current_report.json and submission_manifest.json describe that file; '
                    'the original reference remains in reference_submission.csv.'))
    print('Current recipe export complete. No file was uploaded.')
    return cur_report

current_recipe_report=fit_current_recipe()

# %% cell 98
if CFG.get('run_digit_screen',False):
    s10_replay=load_reproducer('reproduce_digit_study.py')
    s10_replay.run_screen(DATA,output/'recomputed_digit_screen',recorded_digit_study['protocol'])
if CFG.get('run_digit_audit',False):
    s10_selected=recorded_digit_study['screen']['selected_for_final_audit']
    if s10_selected is None:
        raise ValueError('The recorded study selected no recipe for this audit.')
    s10_replay=load_reproducer('reproduce_digit_study.py')
    s10_replay.run_audit(DATA,output/'recomputed_digit_audit',s10_selected,recorded_digit_study['protocol'])
print('Optional method replays never replace submission.csv or upload predictions.')

# %% cell 100
import base64 as s11_base64, zlib as s11_zlib
s11_payload = json.loads(s11_zlib.decompress(s11_base64.b64decode('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')))
recorded_representation_study = s11_payload['study']
representation_reproduction_sources = s11_payload['sources']
for s11_filename, s11_source in representation_reproduction_sources.items():
    (output/s11_filename).write_text(s11_source)
(output/'recorded_representation_study.json').write_text(json.dumps(recorded_representation_study, indent=2));

# %% cell 102
s11_groups = pd.DataFrame([
    {'Group': 'Raw', 'Inputs': 24, 'Representation': 'The original predictors, always retained.'},
    {'Group': 'V', 'Inputs': 24, 'Representation': 'Target encodings of the 24 raw values.'},
    {'Group': 'C', 'Inputs': 12, 'Representation': 'Six numeric cent remainders and six target-encoded cent remainders.'},
    {'Group': 'I', 'Inputs': 36, 'Representation': 'Target encodings of integer, tens, tenths, leading-one, leading-two and length keys for each of six floats.'}
]).set_index('Group')
display(styled_table(s11_groups).set_properties(**{'white-space': 'normal', 'overflow-wrap': 'anywhere'}))

# %% cell 104
def groups_for(full,features):
    cols=list(full.columns)
    groups={
        'V':[c for c in cols if c.startswith('te_value_')],
        'C':[c for c in cols if c.startswith('cent_') or c.startswith('te_digit_cent_')],
        'I':[c for c in cols if c.startswith('te_digit_') and not c.startswith('te_digit_cent_')],
    }
    assert {g:len(v) for g,v in groups.items()}=={'V':24,'C':12,'I':36}
    additions=sum(groups.values(),[])
    assert len(set(additions))==72 and set(cols)==set(features)|set(additions)
    assert len(cols)==96 and not {'id','is_fraud'} & set(cols)
    return groups

def project(matrices,full_cats,features,groups,active):
    allowed=set(features)|{c for g in active for c in groups[g]}
    columns=[c for c in matrices[0] if c in allowed]
    cat_names=[matrices[0].columns[i] for i in full_cats]
    cats=[columns.index(c) for c in cat_names if c in columns]
    result=[a.loc[:,columns] for a in matrices]
    assert len(columns)==24+sum(len(groups[g]) for g in active)
    assert [columns[i] for i in cats]==cat_names
    return result,cats

# %% cell 108
# Pooled-AUC differences and fold differences are distinct quantities.
s11_report = recorded_representation_study['report']
if s11_report['status'] != 'complete':
    raise ValueError('The recorded representation study is not complete.')
s11_recipes = s11_report['recipes']
s11_factorial_keys = ['raw_100', 'V_100', 'C_100', 'I_100',
                      'VC_100', 'VI_100', 'CI_100', 'VCI_100']
s11_sampling_keys = ['V_085', 'VCI_085']
s11_expected_keys = s11_factorial_keys + s11_sampling_keys
if set(s11_recipes) != set(s11_expected_keys):
    raise ValueError('The recorded recipe list differs from the fixed ten-recipe design.')

def representation_label(groups):
    return '+'.join(groups) if groups else 'Raw only'

def representation_delta(with_key, without_key):
    with_recipe, without_recipe = s11_recipes[with_key], s11_recipes[without_key]
    with_folds = {row['fold']: row['auc'] for row in with_recipe['folds']}
    without_folds = {row['fold']: row['auc'] for row in without_recipe['folds']}
    if set(with_folds) != set(range(5)) or set(without_folds) != set(range(5)):
        raise ValueError('A recorded comparison needs all five aligned folds.')
    fold_deltas = [with_folds[k] - without_folds[k] for k in range(5)]
    return {'delta': with_recipe['auc'] - without_recipe['auc'],
            'fold_deltas': fold_deltas,
            'positive_folds': sum(value > 0 for value in fold_deltas)}

s11_result_rows = []
s11_plot_records = []
for s11_key in s11_expected_keys:
    s11_result = s11_recipes[s11_key]
    s11_expected_count = 24 + sum({'V': 24, 'C': 12, 'I': 36}[g] for g in s11_result['groups'])
    if s11_result['feature_count'] != s11_expected_count:
        raise ValueError('A recorded feature count differs from its representation groups.')
    if len(s11_result['folds']) != 5:
        raise ValueError('A recorded recipe has an incomplete fold list.')
    s11_label = representation_label(s11_result['groups'])
    s11_plot_records.append({'key': s11_key, 'label': s11_label,
        'colsample': s11_result['colsample_bytree'], 'auc': s11_result['auc'],
        'fold_aucs': [r['auc'] for r in sorted(s11_result['folds'], key=lambda r: r['fold'])],
        'feature_count': s11_result['feature_count']})
    if s11_key in s11_factorial_keys:
        s11_vs_raw = representation_delta(s11_key, 'raw_100')
        s11_result_rows.append({'Groups': s11_label, 'Inputs': s11_expected_count,
            'OOF AUC': s11_result['auc'], 'Delta vs raw': s11_vs_raw['delta'],
            'Better folds': f"{s11_vs_raw['positive_folds']}/5" if s11_key != 'raw_100' else 'Reference'})
display(styled_table(pd.DataFrame(s11_result_rows).set_index('Groups'),
    {'OOF AUC': '{:.7f}', 'Delta vs raw': '{:+.7f}'}))
print('Every row above is a fresh five-fold recipe with colsample_bytree=1.0.')
print('These are pooled AUC differences; Better folds counts paired fold differences.')

# %% cell 109
"""Plot recorded representation ablations; importing this module performs no fits."""


def representation_factorial_figure(report):
    """Show pooled OOF AUC for the eight fixed full-column representations."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    keys = ['raw_100', 'V_100', 'C_100', 'I_100',
            'VC_100', 'VI_100', 'CI_100', 'VCI_100']
    recipes = report['recipes']
    rows = [recipes[key] for key in keys]
    if any(abs(row['colsample_bytree'] - 1.0) > 1e-12 for row in rows):
        raise ValueError('The factorial figure requires the full-column arm.')
    values = np.asarray([row['auc'] for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('The recorded factorial AUCs must be finite.')
    labels = []
    for row in rows:
        additions = ' + '.join(row['groups'])
        label = 'Raw' + (' + ' + additions if additions else '')
        labels.append(f"{label}  ({row['feature_count']})")

    fig, ax = plt.subplots(figsize=(7.3, 5.7))
    ax.scatter(values, np.arange(len(rows)), s=58, color='#0072B2', zorder=3)
    for position, value in enumerate(values):
        ax.annotate(f'{value:.6f}', (value, position), xytext=(9, 0),
                    textcoords='offset points', va='center', fontsize=13)
    ax.set_yticks(np.arange(len(rows)), labels, fontsize=13)
    ax.set_ylim(len(rows) - 0.45, -0.65)
    span = max(float(np.ptp(values)), 0.0005)
    ax.set_xlim(float(values.min()) - span * 0.12,
                float(values.max()) + span * 0.49)
    ax.set_title('Compare eight representations', fontsize=15, pad=20)
    ax.set_xlabel('Pooled OOF AUC', fontsize=14, labelpad=13)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.4f}'))
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(axis='x', color='#DCE2E6', linewidth=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.85, bottom=0.18)
    return fig


def representation_sampling_figure(report):
    """Compare VCI minus V at the two fixed column-sampling settings."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    recipes = report['recipes']
    comparisons = [('085', 0.85), ('100', 1.0)]
    values = []
    fold_deltas = []
    for suffix, fraction in comparisons:
        base = recipes['V_' + suffix]
        full = recipes['VCI_' + suffix]
        for row in [base, full]:
            if abs(row['colsample_bytree'] - fraction) > 1e-12:
                raise ValueError('The sampling figure requires matched column fractions.')
        base_folds = {row['fold']: row['auc'] for row in base['folds']}
        full_folds = {row['fold']: row['auc'] for row in full['folds']}
        if len(base_folds) != 5 or set(base_folds) != set(full_folds):
            raise ValueError('The sampling figure requires the same five paired folds.')
        fold_deltas.append(np.asarray([full_folds[k] - base_folds[k]
                                       for k in sorted(base_folds)], dtype=float))
        values.append(float(full['auc'] - base['auc']))
    if not np.isfinite(values).all() or not np.isfinite(fold_deltas).all():
        raise ValueError('The recorded paired differences must be finite.')

    fig, ax = plt.subplots(figsize=(7.3, 3.8))
    for position, (pooled, folds) in enumerate(zip(values, fold_deltas)):
        ax.scatter(folds, position + np.linspace(-0.11, 0.11, 5), s=38,
                   facecolors='none', edgecolors='#52616E', linewidths=1.2,
                   label='Individual paired folds' if position == 0 else None,
                   zorder=3)
        ax.scatter([pooled], [position], s=78, marker='D', color='#0072B2',
                   label='Pooled OOF difference' if position == 0 else None,
                   zorder=4)
    ax.axvline(0, color='#A8B3BC', linestyle='--', linewidth=1.1, zorder=1)
    all_values = np.concatenate([np.asarray(values), *fold_deltas, [0.0]])
    span = max(float(np.ptp(all_values)), 0.0005)
    ax.set_xlim(float(all_values.min()) - span * 0.12,
                float(all_values.max()) + span * 0.12)
    ax.set_yticks([0, 1], ['Column fraction 0.85', 'Column fraction 1.00'], fontsize=13)
    ax.set_ylim(1.4, -0.4)
    ax.set_title('Add C and I to value encoding', fontsize=15, pad=18)
    ax.set_xlabel('AUC difference: V + C + I minus V', fontsize=14, labelpad=12)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:+.4f}'))
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(axis='x', color='#DCE2E6', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.43),
              ncol=1, fontsize=12, frameon=False)
    fig.subplots_adjust(left=0.33, right=0.97, top=0.77, bottom=0.40)
    return fig

# %% cell 110
show_figure(representation_factorial_figure(s11_report),         'All eight registered combinations retain the 24 raw predictors. The points show pooled OOF AUC; numbers in parentheses count model inputs. Every tree can use all supplied columns.',         'representation_factorial.png')

# %% cell 112
# Each group is added in all four combinations of the other two groups.
s11_by_groups = {frozenset(s11_recipes[key]['groups']): key for key in s11_factorial_keys}
s11_effect_rows = []
s11_recomputed_effects = []
for s11_group in ['V', 'C', 'I']:
    for s11_without_groups, s11_without_key in s11_by_groups.items():
        if s11_group in s11_without_groups:
            continue
        s11_with_key = s11_by_groups[s11_without_groups | {s11_group}]
        s11_effect = representation_delta(s11_with_key, s11_without_key)
        s11_recomputed_effects.append({'group': s11_group, 'without': s11_without_key,
            'with': s11_with_key, **s11_effect})
        s11_effect_rows.append({'Add': s11_group,
            'Already present': representation_label(s11_recipes[s11_without_key]['groups']),
            'Delta OOF AUC': s11_effect['delta'],
            'Better folds': f"{s11_effect['positive_folds']}/5"})
# Recompute the contrasts from recorded recipe scores, then verify the stored report.
s11_stored_effects = {(row['group'], row['without'], row['with']): row
                      for row in s11_report['conditional_effects']}
if len(s11_stored_effects) != 12:
    raise ValueError('The recorded study needs twelve conditional group contrasts.')
for s11_effect in s11_recomputed_effects:
    s11_stored = s11_stored_effects[(s11_effect['group'], s11_effect['without'], s11_effect['with'])]
    np.testing.assert_allclose(s11_effect['delta'], s11_stored['delta'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(s11_effect['fold_deltas'], s11_stored['fold_deltas'], rtol=0, atol=1e-12)
    assert s11_effect['positive_folds'] == s11_stored['positive_folds']
display(styled_table(pd.DataFrame(s11_effect_rows).set_index(['Add', 'Already present']),
    {'Delta OOF AUC': '{:+.7f}'}))
s11_marginal_rows = []
for s11_group in ['V', 'C', 'I']:
    s11_deltas = [row['delta'] for row in s11_recomputed_effects if row['group'] == s11_group]
    s11_mean_delta = float(np.mean(s11_deltas))
    s11_stored_mean = next(row['mean_conditional_delta'] for row in s11_report['marginal_effects']
                          if row['group'] == s11_group)
    np.testing.assert_allclose(s11_mean_delta, s11_stored_mean, rtol=0, atol=1e-12)
    s11_marginal_rows.append({'Group': s11_group,
        'Mean of four conditional deltas': s11_mean_delta,
        'Smallest conditional delta': min(s11_deltas),
        'Largest conditional delta': max(s11_deltas)})
display(styled_table(pd.DataFrame(s11_marginal_rows).set_index('Group'),
    {column: '{:+.7f}' for column in ['Mean of four conditional deltas',
        'Smallest conditional delta', 'Largest conditional delta']}))

# %% cell 115
s11_verification = recorded_representation_study['measurement_verification']
s11_anchor_checks = s11_verification.get('anchors', s11_verification)
s11_anchor_rows = []
for s11_key, s11_name in [('V_085', 'H8 / V'), ('VCI_085', 'H9 / V+C+I')]:
    s11_original = recorded_representation_study['protocol']['anchors'][s11_key]
    s11_fresh = recorded_representation_study['report']['recipes'][s11_key]
    s11_check = s11_anchor_checks[s11_key]
    s11_difference = s11_fresh['auc'] - s11_original['auc']
    np.testing.assert_allclose(s11_difference, s11_check['auc_difference'], rtol=0, atol=1e-12)
    if s11_check['max_probability_difference'] < 0:
        raise ValueError('A maximum absolute probability difference cannot be negative.')
    s11_anchor_rows.append({'Recipe': s11_name, 'Original OOF AUC': s11_original['auc'],
        'Fresh OOF AUC': s11_fresh['auc'], 'AUC difference': s11_difference,
        'Maximum probability difference': s11_check['max_probability_difference'],
        'Spearman correlation': s11_check['spearman'],
        'OOF .npy bytes equal': 'Yes' if s11_check['byte_identical'] else 'No'})
display(styled_table(pd.DataFrame(s11_anchor_rows).set_index('Recipe'),
    {'Original OOF AUC': '{:.7f}', 'Fresh OOF AUC': '{:.7f}',
     'AUC difference': '{:+.2e}', 'Maximum probability difference': '{:.2e}',
     'Spearman correlation': '{:.8f}'}))
print('The comparison is between frozen historical OOF arrays and newly fitted arrays from this worker, not between submission CSV files.')

# %% cell 117
s11_sampling_rows = []
s11_sampling_deltas = {}
for s11_setting, s11_without, s11_with, s11_stored_key in [
        (1.0, 'V_100', 'VCI_100', 'at_1_0'),
        (0.85, 'V_085', 'VCI_085', 'at_0_85')]:
    s11_effect = representation_delta(s11_with, s11_without)
    s11_sampling_deltas[s11_setting] = s11_effect
    s11_stored = s11_report['sampling_comparison'][s11_stored_key]
    assert s11_stored['without'] == s11_without and s11_stored['with'] == s11_with
    np.testing.assert_allclose(s11_effect['delta'], s11_stored['delta'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(s11_effect['fold_deltas'], s11_stored['fold_deltas'], rtol=0, atol=1e-12)
    s11_sampling_rows.append({'Column fraction': s11_setting,
        'V OOF AUC': s11_recipes[s11_without]['auc'],
        'V+C+I OOF AUC': s11_recipes[s11_with]['auc'],
        'Bundle delta': s11_effect['delta'], 'Better folds': f"{s11_effect['positive_folds']}/5"})
s11_interaction = s11_sampling_deltas[0.85]['delta'] - s11_sampling_deltas[1.0]['delta']
s11_fold_interaction = np.asarray(s11_sampling_deltas[0.85]['fold_deltas']) - np.asarray(s11_sampling_deltas[1.0]['fold_deltas'])
np.testing.assert_allclose(s11_interaction,
    s11_report['sampling_comparison']['difference_of_deltas'], rtol=0, atol=1e-12)
np.testing.assert_allclose(s11_fold_interaction,
    s11_report['sampling_comparison']['fold_difference_of_deltas'], rtol=0, atol=1e-12)
display(styled_table(pd.DataFrame(s11_sampling_rows).set_index('Column fraction'),
    {'V OOF AUC': '{:.7f}', 'V+C+I OOF AUC': '{:.7f}', 'Bundle delta': '{:+.7f}'}))
display(callout(f'The bundle difference at column fraction 0.85 minus its difference at 1.0 is {s11_interaction:+.7f} AUC. '
                'This is a descriptive interaction between the representation bundle and column sampling.'))

# %% cell 118
show_figure(representation_sampling_figure(s11_report),         'The bundle comparison uses fresh V and V+C+I fits at each column fraction. Hollow points are the five paired fold differences; blue diamonds are pooled OOF differences. Fold differences are dependent and are not confidence intervals.',         'representation_sampling.png')

# %% cell 122
s11_run = recorded_representation_study['report']
s11_protocol = recorded_representation_study['protocol']
if s11_run['actual_fits'] != s11_protocol['fit_budget'] or s11_run['actual_trees'] != s11_protocol['tree_budget']:
    raise ValueError('The completed study does not match its registered fit/tree budget.')
s11_start = pd.Timestamp(s11_run['started_utc'])
s11_end = pd.Timestamp(s11_run['completed_utc'])
if s11_end < s11_start:
    raise ValueError('The recorded study completion precedes its start.')
s11_wall_minutes = (s11_end - s11_start).total_seconds() / 60
s11_run_facts = pd.DataFrame({'Recorded execution': {
    'Protocol fixed (UTC)': pd.Timestamp(s11_protocol['created_utc']).strftime('%Y-%m-%d %H:%M:%S'),
    'Study started (UTC)': s11_start.strftime('%Y-%m-%d %H:%M:%S'),
    'Study completed (UTC)': s11_end.strftime('%Y-%m-%d %H:%M:%S'),
    'Completed predictive fits': str(s11_run['actual_fits']),
    'Actual fitted trees': f"{s11_run['actual_trees']:,}",
    'Study wall time (minutes)': f'{s11_wall_minutes:.2f}'
}})
display(styled_table(s11_run_facts))
s11_environment = pd.DataFrame({'Recorded version': s11_run['libraries']})
display(styled_table(s11_environment))
print('These are the recorded study worker timings and versions. Study wall time includes preparation and file operations.')

# %% cell 124
if CFG.get('run_representation_study', False):
    from datetime import datetime, timezone
    s11_runner = load_reproducer('representation_study.py')
    s11_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    s11_replay_output = output/'recomputed_representation'/s11_stamp
    s11_recomputed = s11_runner.run(DATA, s11_replay_output,
        recorded_representation_study['protocol'])
else:
    print('The representation study is a recorded snapshot. Its 50-fit replay is disabled.')
print('This study does not change the primary CSV, the current model or submission decisions.')

# %% cell 126
import base64 as s12_b64, zlib as s12_zlib
s12_payload=json.loads(s12_zlib.decompress(s12_b64.b64decode('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')))
recorded_shallow_study=s12_payload['record']
shallow_reproduction_sources=s12_payload['sources']
shallow_source_dir=output/'shallow_refit_source'
shallow_source_dir.mkdir(exist_ok=True)
for s12_name,s12_text in shallow_reproduction_sources.items():
    (shallow_source_dir/s12_name).write_text(s12_text)
(output/'recorded_shallow_study.json').write_text(json.dumps(recorded_shallow_study,indent=2));

# %% cell 128
s12_report=recorded_shallow_study['report']
s12_rows=[]
for s12_row in s12_report['true_comparisons']:
    s12_single=s12_row['single_models']
    s12_rows.append({'Outer split':s12_row['split_seed'],
        'H9 single':s12_single['h9']['auc'],'H10 single':s12_single['h10']['auc'],
        'H10 bag':s12_row['h10_bag_auc'],'H10 - H9':s12_row['h10_minus_h9'],
        'Single - bag':s12_row['single_minus_bag']})
s12_table=pd.DataFrame(s12_rows).set_index('Outer split')
display(styled_table(s12_table,{'H9 single':'{:.7f}','H10 single':'{:.7f}',
    'H10 bag':'{:.7f}','H10 - H9':'{:+.7f}','Single - bag':'{:+.7f}'}))
print('Every AUC above is on an outer holdout. These are not pooled five-fold OOF scores.')

# %% cell 129
"""Display the two predeclared contrasts; no fitting or inference test."""
fig,axes=plt.subplots(2,1,figsize=(7.3,7.5))
for ax,key,title,gate in zip(axes,['h10_minus_h9','single_minus_bag'],
        ['Compare the fixed H10 and H9 recipes', 'Compare one larger fit with five smaller fits'],
        [.0005,.0001]):
    values=[r[key] for r in s12_report['true_comparisons']]
    labels=[str(r['split_seed']) for r in s12_report['true_comparisons']]
    ax.scatter(values,range(2),color='#0072B2',s=65,zorder=3)
    ax.set_yticks(range(2),labels,fontsize=13)
    ax.set_ylim(1.6,-.6)
    ax.axvline(0,color='#526271',linewidth=1)
    ax.axvline(gate,color='#c47e2d',linestyle=':',label=f'Fixed gate: {gate:.4f}')
    ax.set_title(title,fontsize=15,pad=15)
    ax.set_xlabel('Held-out AUC: H10 minus H9' if key=='h10_minus_h9'
                  else 'Held-out AUC: single 80% minus bag of five 64% fits',
                  fontsize=13,labelpad=10)
    ax.set_ylabel('Outer split seed',fontsize=13)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x:.4f}'))
    ax.tick_params(axis='x',labelsize=12)
    ax.grid(axis='x',alpha=.15)
    low=min(0,min(values));high=max(gate,max(values));pad=max(.0003,(high-low)*.65)
    ax.set_xlim(low-pad,high+pad)
    for i,v in enumerate(values):
        ax.annotate(f'{v:+.7f}',(v,i),xytext=(8 if v>=0 else -8,0),
                    textcoords='offset points',va='center',ha='left' if v>=0 else 'right',fontsize=13)
    ax.legend(loc='lower right',fontsize=11,frameon=True,facecolor='white',
              edgecolor='white',framealpha=1)
fig.subplots_adjust(left=.25,right=.96,top=.90,bottom=.11,hspace=.75)
show_figure(fig,
    'The upper panel compares H10 with H9 on the same fitting and held-out rows. '
    'The lower panel compares one H10 fit on 80% of all rows with five smaller fits on 64% each. '
    'Each point is a paired difference on one outer holdout. The dotted lines mark fixed operational gates; '
    'the overlapping holdouts are not independent replications.',
    'shallow_refit_comparisons.png')

# %% cell 132
s12_review=pd.DataFrame(recorded_shallow_study['review_budget']['rows'])
s12_review_table=s12_review.pivot(index='reviewed_rows',columns='recipe',values='frauds_found')
s12_review_table=s12_review_table.rename(columns={'h9_digits':'H9 frauds found','h10_small_trees':'H10 frauds found'})
s12_review_table.index.name='Transactions reviewed'
display(styled_table(s12_review_table))
print('The full training target contains 11,689 fraud cases. Fractions: 0.5%, 1%, 2% and 5%.')

# %% cell 135
s12_control_rows=[]
for s12_kind,s12_values,s12_family in [('Negative',s12_report['negative_controls'],6),
                                    ('Target-copy canary',s12_report['canaries'],2)]:
    for s12_r in s12_values:
        s12_control_rows.append({'Control':s12_kind,'Outer split':s12_r['split_seed'],
            'Seed':s12_r.get('label_seed',2521),'AUC':s12_r['auc'],'MC p':s12_r['p'],
            'Adjusted p':min(1.,s12_family*s12_r['p'])})
display(styled_table(pd.DataFrame(s12_control_rows).set_index(['Control','Outer split','Seed']),
    {'AUC':'{:.7f}','MC p':'{:.4f}','Adjusted p':'{:.4f}'}))
print('Combined new audit gate:',s12_report['new_audit_pass'])
print('Full-training policy proxy gate:',s12_report['refit_proxy_pass'])
print('Actual predictive fits:',s12_report['fit_count'],'of at most 23.')

# %% cell 140
s12_public=pd.DataFrame(recorded_shallow_study['exports']).set_index('Export')
display(styled_table(s12_public,{'CV recipe OOF':'{:.7f}','Accepted public':'{:.5f}'}))
print(recorded_shallow_study['selection_statement'])

# %% cell 144
if CFG.get('run_shallow_refit_study',False):
    from datetime import datetime, timezone
    import importlib.util as s12_import, sys as s12_sys
    s12_stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    s12_work=output/('shallow_refit_replay_'+s12_stamp)
    s12_work.mkdir(exist_ok=False)
    for s12_name,s12_text in shallow_reproduction_sources.items():
        (s12_work/s12_name).write_text(s12_text)
    s12_old_path=list(s12_sys.path)
    s12_old_core=s12_sys.modules.pop('digit_core',None)
    try:
        s12_sys.path.insert(0,str(s12_work.resolve()))
        s12_spec=s12_import.spec_from_file_location('s12_fixed_runner',s12_work/'shallow_refit.py')
        s12_module=s12_import.module_from_spec(s12_spec);s12_spec.loader.exec_module(s12_module)
        s12_protocol=json.loads((s12_work/'protocol.json').read_text())
        s12_module.preflight(DATA,s12_protocol,s12_work/'preflight')
        s12_module.run(DATA,s12_work/'results',s12_protocol)
    finally:
        s12_sys.path[:]=s12_old_path
        s12_sys.modules.pop('digit_core',None)
        if s12_old_core is not None:s12_sys.modules['digit_core']=s12_old_core
else:
    print('The fixed shallow/refit study is disabled. Its recorded results are shown above.')

# %% cell 146
import base64 as s13_b64, zlib as s13_zlib
s13_payload=json.loads(s13_zlib.decompress(s13_b64.b64decode('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')))
recorded_seed_study=s13_payload['record']
seed_reproduction_sources=s13_payload['sources']
seed_source_dir=output/'seed_bag_source'
seed_source_dir.mkdir(exist_ok=True)
for s13_name,s13_text in seed_reproduction_sources.items():
    (seed_source_dir/s13_name).write_text(s13_text)
(output/'recorded_seed_study.json').write_text(json.dumps(recorded_seed_study,indent=2));

# %% cell 148
s13_report=recorded_seed_study['report']
s13_member_rows=recorded_seed_study['member_rows']
display(styled_table(pd.DataFrame(s13_member_rows).set_index('Predictor'),
    {'OOF AUC':'{:.7f}','Delta vs seed 42':'{:+.7f}'}))
s13_fold_table=pd.DataFrame(recorded_seed_study['fold_rows']).set_index('Fold')
display(styled_table(s13_fold_table,
    {'Seed 42':'{:.7f}','Three-seed average':'{:.7f}','Average - seed 42':'{:+.7f}'}))
print('The pooled OOF score is not the mean of the five fold scores. Fold differences are dependent.')

# %% cell 149
s13_deltas=[r['Average - seed 42'] for r in recorded_seed_study['fold_rows']]
s13_pooled=recorded_seed_study['report']['average']['delta_vs_baseline']
s13_values=s13_deltas+[s13_pooled]
s13_labels=[f"Fold {r['Fold']}" for r in recorded_seed_study['fold_rows']]+['Pooled OOF']
fig,ax=plt.subplots(figsize=(7.3,5.8))
ax.scatter(s13_values[:5],range(5),s=55,color='#526271',zorder=3)
ax.scatter([s13_pooled],[5],s=90,marker='D',color='#0072B2',zorder=3)
ax.axvline(0,color='#526271',linewidth=1)
ax.axhline(4.5,color='#cbd5df',linewidth=1)
ax.set_yticks(range(6),s13_labels,fontsize=13)
ax.set_ylim(5.7,-.7)
ax.set_title('Average three fixed model seeds',fontsize=15,pad=15)
ax.set_xlabel('OOF AUC: equal average minus seed 42',fontsize=13,labelpad=10)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x:.4f}'))
ax.tick_params(axis='x',labelsize=12)
ax.grid(axis='x',alpha=.15)
lo=min(0,min(s13_values));hi=max(0,max(s13_values));pad=max(.00015,(hi-lo)*.55)
ax.set_xlim(lo-pad,hi+pad)
for i,value in enumerate(s13_values):
    ax.annotate(f'{value:+.7f}',(value,i),xytext=(8 if value>=0 else -8,0),
                textcoords='offset points',va='center',ha='left' if value>=0 else 'right',fontsize=12)
fig.subplots_adjust(left=.23,right=.96,bottom=.15,top=.89)
show_figure(fig,
    'Each fold point subtracts seed 42 from the equal probability average of seeds 42, 137 and 271. '
    'The blue diamond recomputes AUC after pooling every OOF row. It is not the mean fold difference. '
    'The reused folds overlap in training and do not provide independent replications or confidence intervals.',
    'seed_bag_comparisons.png')

# %% cell 152
display(styled_table(pd.DataFrame([recorded_seed_study['gate_summary']]).set_index('Comparison'),
    {'Minimum pooled gain':'{:+.7f}','Measured pooled gain':'{:+.7f}'}))

# %% cell 159
display(styled_table(pd.DataFrame(recorded_seed_study['exports']).set_index('Export'),
    {'CV recipe OOF':'{:.7f}','Accepted public':'{:.5f}'}))

# %% cell 162
if CFG.get('run_seed_bag',False):
    from datetime import datetime, timezone
    import importlib.util as s13_import, sys as s13_sys
    s13_stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    s13_work=output/('seed_bag_replay_'+s13_stamp)
    s13_work.mkdir(exist_ok=False)
    for s13_name,s13_text in seed_reproduction_sources.items():
        (s13_work/s13_name).write_text(s13_text)
    s13_old_path=list(s13_sys.path)
    s13_old_core=s13_sys.modules.pop('digit_core',None)
    try:
        s13_sys.path.insert(0,str(s13_work.resolve()))
        s13_spec=s13_import.spec_from_file_location('s13_fixed_runner',s13_work/'seed_bag.py')
        s13_module=s13_import.module_from_spec(s13_spec);s13_spec.loader.exec_module(s13_module)
        s13_protocol=json.loads((s13_work/'protocol.json').read_text())
        s13_module.preflight(DATA,s13_protocol,s13_work/'preflight',rebuild_reference=True)
        s13_module.run(DATA,s13_work/'results',s13_protocol,rebuild_reference=True)
    finally:
        s13_sys.path[:]=s13_old_path
        s13_sys.modules.pop('digit_core',None)
        if s13_old_core is not None:s13_sys.modules['digit_core']=s13_old_core
else:
    print('The 15-fit seed-average replay is disabled. The recorded 10-fit study is shown above.')

# %% cell 164
import base64 as s14_b64,zlib as s14_zlib
s14_payload=json.loads(s14_zlib.decompress(s14_b64.b64decode('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')))
recorded_glm_study=s14_payload['record']
glm_sources=s14_payload['sources']
glm_source_dir=output/'joint_glm_source'
glm_source_dir.mkdir(exist_ok=True)
for name,text in glm_sources.items():(glm_source_dir/name).write_text(text)
(output/'recorded_glm_study.json').write_text(json.dumps(recorded_glm_study,indent=2));

# %% cell 166
s14=recorded_glm_study
s14_rows=[{'GLM weight':x['glm_weight'],'OOF AUC':x['auc'],'Gain vs H10':x['delta'],'Positive folds':x['positive_folds'],'Role':x['role']} for x in s14['comparisons']]
display(styled_table(pd.DataFrame(s14_rows).set_index('GLM weight'),{'OOF AUC':'{:.7f}','Gain vs H10':'{:+.7f}'}))
s14_base=s14['comparisons'][0]
s14_candidate=s14['candidate']
s14_folds=pd.DataFrame({'H10':s14_base['fold_aucs'],'80/20 mixture':s14_candidate['fold_aucs'],'Mixture minus H10':s14_candidate['fold_deltas']})
s14_folds.index.name='Fold'
display(styled_table(s14_folds,{'H10':'{:.7f}','80/20 mixture':'{:.7f}','Mixture minus H10':'{:+.7f}'}))

# %% cell 167
fig,ax=plt.subplots(figsize=(7.3,3.9),dpi=100)
s14_d=s14_candidate['fold_deltas']+[s14_candidate['delta']]
for i,v in enumerate(s14_d):
    ax.scatter(v,i,color='#2775a3' if i==5 else '#65717d',s=65 if i==5 else 30,marker='D' if i==5 else 'o')
    ax.annotate(f'{v:+.7f}',(v,i),xytext=(7 if v>=0 else -7,0),textcoords='offset points',ha='left' if v>=0 else 'right',va='center',fontsize=12)
span=max(abs(v) for v in s14_d) or .0001
ax.set_xlim(-span*1.75,span*1.75);ax.axvline(0,color='#97a1ab',lw=1)
ax.set_yticks(range(6),[f'Fold {i}' for i in range(5)]+['Pooled OOF']);ax.invert_yaxis()
ax.set_xlabel('OOF AUC: fixed 80/20 mixture minus H10');ax.set_title('Does an additive model help shallow trees?',loc='left',fontsize=15)
ax.tick_params(axis='both',labelsize=12)
ax.spines[['top','right']].set_visible(False);fig.tight_layout()
show_figure(fig,'The mixture uses 80% H10 and 20% logistic probabilities. The pooled point recomputes AUC across all OOF rows; it is not the average fold difference. These reused folds are dependent.','glm_comparison.png')

# %% cell 170
s14_control_rows=[]
for kind,entries in [('Shuffled labels',s14['report']['controls']),('Copied target (invalid)',s14['report']['canaries'])]:
    for x in entries:s14_control_rows.append({'Check':x['name'],'Kind':kind,'AUC':x['auc'],'Adjusted p':x['bonferroni_p'],'Solver iterations':x['n_iter']})
display(styled_table(pd.DataFrame(s14_control_rows).set_index('Check'),{'AUC':'{:.7f}','Adjusted p':'{:.4f}'}))

# %% cell 173
display(styled_table(pd.DataFrame([recorded_glm_study['gate_row']]).set_index('Candidate'),{'OOF gain':'{:+.7f}','Required gain':'{:+.7f}'}))

# %% cell 179
if CFG.get('run_glm_study',False):
    import sys as s14_sys,importlib.util as s14_import,psutil as s14_psutil
    from datetime import datetime as s14_dt,timezone as s14_tz
    if CURRENT_CFG!=s14_payload['baseline_cfg'] or not CFG.get('run_current_recipe',True):
        raise ValueError('This fixed replay requires the original H10 CV settings and current-model run.')
    if s14_psutil.virtual_memory().available<8_000_000_000:
        raise MemoryError('The sparse replay requires at least 8 GB available memory.')
    s14_work=output/('joint_glm_replay_'+s14_dt.now(s14_tz.utc).strftime('%Y%m%dT%H%M%S%f'));s14_work.mkdir()
    for name,text in glm_sources.items():(s14_work/name).write_text(text)
    s14_path=list(s14_sys.path);s14_previous=s14_sys.modules.pop('digit_core',None)
    try:
        s14_sys.path.insert(0,str(s14_work.resolve()))
        s14_spec=s14_import.spec_from_file_location('s14_fixed_glm',s14_work/'joint_glm.py');s14_module=s14_import.module_from_spec(s14_spec);s14_spec.loader.exec_module(s14_module)
        s14_protocol=json.loads(glm_sources['protocol.json'])
        s14_module.preflight(DATA,s14_work/'preflight.json')
        s14_report=s14_module.run(DATA,s14_work/'results',s14_protocol)
        if s14_report['status']=='complete':
            s14_g=np.load(s14_work/'results/oof.npy');s14_gt=np.load(s14_work/'results/test.npy')
            s14_h=np.load(output/'current_oof.npy');s14_ht=np.load(output/'current_test.npy')
            s14_current=json.loads((output/'current_report.json').read_text())
            assert s14_current['status']=='complete' and s14_current['current_settings']==s14_payload['baseline_cfg']
            assert s14_current['export_mode']=='cv_average'
            assert s14_current['output_oof_sha256']==s14_module.sha(output/'current_oof.npy')
            assert s14_current['output_test_sha256']==s14_module.sha(output/'current_test.npy')
            s14_folds=pd.read_csv(output/'current_folds.csv');s14_ids=pd.read_csv(DATA/'train.csv',usecols=['id','is_fraud'])
            np.testing.assert_array_equal(s14_folds.id,s14_ids.id)
            np.testing.assert_array_equal(s14_folds.fold,np.load(s14_work/'results/fold_ids.npy'))
            assert s14_h.shape==s14_g.shape==(len(s14_ids),) and s14_ht.shape==s14_gt.shape
            assert all(np.isfinite(x).all() and ((x>=0)&(x<=1)).all() for x in [s14_h,s14_ht,s14_g,s14_gt])
            s14_mix=.8*s14_h+.2*s14_g;s14_mix_test=.8*s14_ht+.2*s14_gt
            s14_sample=pd.read_csv(DATA/'sample_submission.csv');s14_test_ids=pd.read_csv(DATA/'test.csv',usecols=['id'])
            s14_order=pd.Index(s14_test_ids.id).get_indexer(s14_sample.id);assert (s14_order>=0).all()
            s14_sample['is_fraud']=s14_mix_test[s14_order];s14_sample.to_csv(s14_work/'mixture_submission.csv',index=False)
            np.save(s14_work/'mixture_oof.npy',s14_mix);np.save(s14_work/'mixture_test.npy',s14_mix_test)
            s14_y=s14_ids.is_fraud.to_numpy();s14_gain=float(roc_auc_score(s14_y,s14_mix)-roc_auc_score(s14_y,s14_h))
            s14_deltas=[float(roc_auc_score(s14_y[s14_folds.fold==k],s14_mix[s14_folds.fold==k])-roc_auc_score(s14_y[s14_folds.fold==k],s14_h[s14_folds.fold==k])) for k in range(5)]
            s14_result={'status':'complete','auc':float(roc_auc_score(s14_y,s14_mix)),'gain':s14_gain,'fold_deltas':s14_deltas,'operational_gate':bool(s14_gain>=.00005 and sum(x>0 for x in s14_deltas)>=4),'historical_eligibility_inherited':False,'submitted':False,'primary_unchanged':True,'sha256':{n:s14_module.sha(s14_work/n) for n in ['mixture_oof.npy','mixture_test.npy','mixture_submission.csv']}}
            (s14_work/'mixture_report.json').write_text(json.dumps(s14_result,indent=2))
            print('Fresh fixed-mixture OOF:',s14_result['auc'])
        print('Replay status:',s14_report['status'],'; primary submission.csv is unchanged.')
    finally:
        s14_sys.path[:]=s14_path;s14_sys.modules.pop('digit_core',None)
        if s14_previous is not None:s14_sys.modules['digit_core']=s14_previous
else:
    print('The optional GLM replay is disabled. The primary export remains the model from section 17.')

# %% cell 181
import base64 as s15_b64,zlib as s15_zlib
s15_payload=json.loads(s15_zlib.decompress(s15_b64.b64decode('eNrNfflzG8uR5r/SwQkb5BMI1n1Qi51w2ONdb8za3rFn9wc+BqLR3SAxAgEYDUii32r/9v2yGlcfRRKCNPKzgwL6qKrM/Cqvyir8crEqssUqv7j95SJbFem6yEebdXZxeyGYMNecXzPxVy5uJbtVfiCYEtq9Y+yWsYv+RblO15sSz2aLp+WsWBe4tlwt1otsMRuVj6nQBje1lkUhC21S44WTrNATNJ0XmcgnaE2pgiufyVTpzGS+4IXF94Jrw/PMMZ9SR9lqulwfmnQTPh4rN1ZW8QnLOJpS2ipTWC1YoXg2UblXhcryNC/UWBjrpRJjLguWZ5mdoEkQXRwaNFxmuSucYRM7kSLFyBj3jBc6TQvBJqlNrbI8M6l0jAmlJxggY3nqx5bjGhqczpeb9egxLR+Lkrg5S8fFrByl83w0zcOVZbp+RF95uk5v1qt0Oh9k5Ueibk+WLdKUs7HLOfor1MRoiy8p+jVCCSW5lG6c525sUhDsDNeCpxnLWOEn41xffOlfPHI2WiwmR/2tirJIV9njiCTKORM3ZQEpj9OHm1UxKVbFPCtu8Mpgvnw+Hs3YOW1TxowTWc75uNAqFV45zlI1SYtiLFNZTJwYs2Ji9USbCbFJjXPwmXtnaDSTxaxJ/puGs3uvOSavjDZWajkxLuMG/FITn6ZM2jTNmXLW+Nw6SD/LMHBvPc9ymaXMSl5kQBiN6WH21OBQ9lhkH8obwPkBWLxZrqYfMRFGi80aIr2h5zHozWxddrFJ5XYCQXAMoUizXBvlTM5dlmZajeUkz6yRTgnJXAqOOonxjw3AlFvmJpBtYNP0M7FgVgAsLw4tA5ymAFCnwDy3YEhu89QKpySzrHAQRDGRupBGmFR5PzGpS623kgE7Oi0Y3uUTIzNhwBwMZbX4BGkZ771hGNgq3eSj9WKdzi5uOTfO9y+2vLi4vfvlYrzJH4r1xS0bMI65v5hNs2eMZLlYzIr8glqbFfheFvNyup5+nK5pwIAoUfh6T2UxKzJopItbyxieSbP1Jp2N8Gi2ni7mVbe4wzRUhuFCGKn1rqnDy8oovDwrHjCEJxLs4ZaQ1PcSOnBablvEVHPU3u4vEZylsxndk954rRR4B8UjvJZ12f0nU6V5lCrlW1QpdUSXrlNlHcjSFnpSc6MMqMqL2TodHZ7BcLziTGvjoYOgyK3aPVTrhxFvDswzxu1GDthkGxqdNrtLs0UJ7CgMZl6sR9W1B+hFYlWNtDTP6U2jaldXxdPiY+AjWvjbptgUo+l8XazKYstI4xzb3dnMwyXLxf7Sf6RZlpLpYwMPqXvBpXISPGcW9mGzXkwmo/W00uZbzAZuNIXcFGj1auCsYVDSaBGgkbiJdtFgPkrXo91D/PB+7fKXNrRO7hxaSXGnoWad0v6Uvrequ5rk9AkNNiYulxiBaEBcSGmPx8TD/OpCOPMAiFPMKQe9rUwb4J65CL6VFi14W6Ms084JqWGkavCGHZPQv0ZaoFdLE5u0b6TIyhhFjlnLpIB2lVxY20ESV1GS2jOWRqsxj6xiltdJ8px5D4GDZoDcd05YQZbSK8xb72F1fMd8vQ4T1mv4Mpj3Hi4FqNCtGUudH89Y156wpmu+CtY9X7nsnq9wb3xjvkImqnO+Migs7qVxVkj+pb9DKX8jSl1NptCNMT0sGaEK3QE/XHXJlN7tlKlmLdsCIGoMmkN40Ov8WKaKCSBegSALuQmhT4VpgySpYyQJDZJgwazVlh1ke0QSiaEbpt60SEJPWpG3ZZwhXhyTxI0AyczCYnirWjCt8CdI6VrvMG+gfxWPGRYGqZPt8XComDyYkT1OG4bFt2F6Lbtw6k+EqRJNs8JhPbtgipiEabiryhjydg84FV+HUx7FKQwN8GMwKeDdiQ5/AZrv7brHKi/gTjroF6GVrwvVIlqSSkG5MC/FmTAl/HVTxMkDgvbTDkIXXQSZKEGuRZABejhiF0WmojHxoEGh/QS8Zicc053a9M0oRVRCuhf2SHDBjXlVm5Kv08RpJ0zJa+vEqYmpU6dbOKWBd6hTjodhwLSCD+jsAafy63DK4jiF1dfWew6rLMiDbevTmFurSG4NsXK4NkwIh5BcGtXQp3AtrBbWwSdV58FUkt7rNvrwY7x31Apz2nT5MUpHcdomCP41TLYl8DCp6wQp7rkBvuDMg/JumFoOTSARfksuMT1tBKbawjpZzHyD+UCCb8OU12Fq2yhVnSjl3SgVPIZSq1ooFd1GXzlE+OADXBXpxAGl6qtQirA86slB41gEBBoeGELotlCdjyof11I+pCodInJPTqiuR5QOrikMhIDYueTWnIlTJ6IkEU6FwayD/WSiiyQbJUm3SPIMjoxUwjtMQdsgySBsVQ7kRlDKFNxjMNcbbR08Ih0NJmE30ZWEQYJqDjP8FZhy1hFNnmL0RQymRrRgyjphCv0PRbr7y7/cE1BjmZJiNQoY/ofJlUgTz5Wodq7kOFkiG7kSdAPMaThtzP7QTIlSUZp0y6E18qVMCWYPvFC0iCA0Am7jFOlqRd4YXF8d9RSO+hFCinamRDcyJR062HUmSmQkUcKjiRLRSpRw150oAXUIYAQMiKVQuDtR0qGMa/5BPVOhEaQqxHFQTPDyT0qT9C/4SV1BFVt4OdLBq4R/q07qSpzUlYHGR/wuJC2a6NN6kqf1xDUhWQoYcwDTnNSVOk1UiGgQYsCzUAiNpTkro3UiSgycU0RRTBvNg036fijhUmkNpgKPSvnvChKHkIMBkl4DkPJ7ogTKSMLRRxijqsD/+6FEA5IORh4+vlL6h6Qe1Rkmw4uoz25bvpBh9liV1w2Gdsxoii8VHEAnz8s9nkcSi7p37TxVjSJeJ0mGkAFqRhsndHdSR1DiDqbfUsQCB1fFso+y1pV2tmUFbcPBEx1pHd0ZicRcPHVKJGI6rSDldKB2nQZxQp6df/zHkKuiIIEiBEoS2jMTO2eRFA2ubNsF5S+4awaTb5+6iabJDXSvt14ZjzHFImZV83RlG6e+jlP5xiR5DKQ6mtRpxyHdrpolFnPhrWFCaHt28vEciSp/Qg7ERhdgFaLOo7yO+IEYVSesTtnatGvkyDWZYwO7YjDvKPfWpU9rEbPkcBhiCrVmjKCp5esJyI4F2OvOmJmLCFj599Oo8geANapR29UCdYXazEAyLSl54WDl9I9UqC+kIFvujD096ygo1QwF64TzGKiL6VBZrw3w7tV0jmlDU5+ETBtFpm0jU3Yik9bjqHJNUChgz806niVHJqILc+2sYw2Yop6i20sLUQFXPxCZLIpM3046mripd5Y5MnWeSSmNOGsNR9QzM+obLuHwk13S9hKO0N3JcQ8uwzIiPEfM1co6ihfqs5YrjGf1lflGkn8EAIJVaS4PJsJgd6z2UNVGBAAWurOBAGVqouG1hKPWUtNykIPfY5zy52Ucz6TK8DhVvkVVHcsajcKl87QwqbQXEY8AWPdaIbgk5w5GM+oQ1GYN523PVbFGnrFr5dx3V3i4CKC9jyNadKz3dCcbNXeOkc8DbWD9V1ZlbUV5SE9wyx2DKjBCMCGN/K5lWR29C8xQrWkpFFTRctF/fnZEhEXwTqVNpSmaCo/g7cBZk214c2FiCXWQZJvwBo1MI2YCEDmre78aGDdKahghTitJZ6VHhKW1tQhN6ADTxFLNtBedJOk4SbJFkgvlWd7SwjNndT1EpaHEQsTQjksVXQFz3jOu4VfADsuIKWJUeqThd8FcSRfqP15Oj5g3Ln81lnn2E1ex7nkrLG9aIhib7mVaWtJDTM2DwXVnJkeEiy3TCkawsZhO6Aj9dSLVxwJPcF63xApn1zBaZAYipWxoZbrGLNXkkYU9y2eC9+tiRCncg/kyyqEjqTqJ4lGibHv6McrFWVb90zA0HAjVlKrVKmZopIK+ElRVZKiS1MYNDZQKPaA52Ce1atcUyNfdpuvOBS0ZqfxVkdVacvSaaPWiu/SFSlGMpPI7eMxnJklEyOB3qyAHFjIYdqrI5axLrpK7qA4yLbBaSXllKFaI0ahaLg/sp4VK6AwfSvDOw6olz6ibJtJzVO8kEUtY30lSdP4Fz69BEoTkpdGwyfhQxypI1IjDPKaHhdV4va7g5SwJFeRBt3oqZcO86PDx5et5527NGnGJpIxpVulamlV2a1YEUtpA+8DHh9k8M0mCOR0Fq4crTUWdmpBnXadkZVSyri1ZKqWFSNAkTBqVSR2LFiLjuK/hHdjTnYCGZo3VSAoypg4RBPwQGkUnTSyqWJloue2ajLO3tMyrlKs7Ace2AnD++tLX6yr5rEKdELwORAbCvJZ87lwk6YxITaRG+4DJJlqVbKH1UBpzjFbtlKBiCMp+Wc/PTJxgZkfVkANaMY0x3/GXmy6TaU1csm33zkPXQFtTFayTrK6I4M/CviFgxAARy5ypW23cXjAHZsOnhSve7YXHaerwwuEr0WSGJ46gjblXSGrp1ph/6mnZiDYRMG8t823/1L26d4B14VJFcKlMDJei7Z8K251q9pxGDdVP+yJezpS06rN+aK5ERrMKoi1wUU+3NnIl1jiEoPCPEeO5H5sr8fFcCW9TxV4gSlEc4ATFHiqic8mfRdjPaKeO0PAzVNyfPc6fWsfaToJqbDnQHdWH16o7cxJxaQVj8cwJ60gGsjdkA7+qTAuv1xMXJkS7QhAqXKgP+1YFOK2uLJXq01ZLQ59OrAgTp3VFGw0QQjqIhBn/DStwOrqCXw5la2kzWliB+FYVOG1Z0UKMIRUI94Kr71in1QET+NPawgVSTujvCRILgMCps1RbzU6snToVJIZSYpQFEd8XImievCYnvTVWy+8IEWjAsJVXC0UQ+SFpSPf1hoOHPcgRn06Ylk+nGy4PVXEoQ1xg0p2XePwHouKNjttre+fMG900KU8Ndg1vB7u+s3wFwCQ/hSNKt4KfnUY8R0YumhzWvJ1FdD5WEqCVN5SW2IWHZ4a6Z5EUhx1vw87FXUl4Gs7Kfa3V2fUrNa9VSmNfw6p9a/mKtBGwRmMKbVtgVd1FAh7OmaCNqRpwPjeLeI5gX8q4uZY7DecnKth2xu0HYRWOejyJ6NpJxHhJjqaTGA4VV/a8rZ61UittOvbQNZDK5RvrBFSkTkCpE7Iy3VuSTTiogHZV0TlG56YQz0PqCxnEdjj7glb9tgnEc0hi8cnn27GsbawweeeYkF4L5rXjkZxhbYdnqAaNnOtQq5Tw7NXUTGfSsDN41TF42ig821Zf++6koRWIM0M1pKOjaM5LGp5nIkXcRLqOVe5adXXdRsKQcb/fOf8jzX48wy1Ym6YXym1qZasuZvehGBVI97D5RnPP31S2aoXwbWXK3gBX0alNxalr3aq9IqPFW8pWm7lE/T1PxZLK+8gyckepndMsemCUaSeMBQThvab0oKhi3oPsoWXpFXRPh+1YcV768GQy7Atk8BYZtAuZSfhmDIJiTTosbUCW8NqUjGznJMkyyRlNYsecj2XCOe0LcPBXPTO2qlpoZgsbxa1Sv3GPgGERBIdlrC4I02pIMx8utYmc/iA9RzDi6Xga5tRXllltxXhIMDB4j4IZcE+hB31ahuHUMquu3oFQQY4OpOFPyg59s/wG7WXoRrZyCkG7Fiqk4LvOv+JGR08W4q3MAHBuhacEIxxXa+prVtUpURxWGCD2p4eddZoc4zGaNBU8WQpw6eCWLpqihZHAB2vRpIRwFsLDzFOsFp1Q4ALegUfw0vFE59yVUMvMk0kECoRlsa3YAEkwzgxRrNLKvF4NoNoz13cX/EbqrDiLlK4YY5v1kSZsie4qj/QWAY3WUiJmP7fQirZ+RbQwF0Cp0M5RvOW7SleMi5Wu6HY1gJCUNGDdtekI3zwcPybgxHBx5vlCNhzA001TKPvxiBkc/KZOrDobpcn6Dpq4FQixOeHINWlCfEiHighajfHn5EjsAI3TAXOKDjXCZC+umX61HoC9fYtP7JQhEcmS0EJa8/AAySLeEsIbR+ec0eEq5xZboeNIRYBmAlCFVtWIk4CkLu1qo3lXLVrFVvCAEC3DptKWCcpQ1aSLmMU4NEjVAlqdd4QLVLWOEUX7b+hMQKGpALWTJhOlybUQq+nMJyotMcIpXafIGzoaFcOViCgijtGbXXsDziF4J0fS0a761wLRjuSzOA2rLHbQhW9jldturFIZGjxDr6Vlzp2ZKKFakphYCVUWU8ZSuUeXVKM7fOncoLbNBMPhvCMaxBSoA5UOHw6HgnLI3LozgapYfPbBE7BUZcG8OZmkDjdAcnf4n3iFpjfXrkiOEXqoUCYtPtpvVbsSNhF3nyhoYrhkuoVLpyJHtRlp4OmROrDnVlUZbyLV1YgIVShx1zBLvFOKxkQNpGjHk7C1tI8OsZSg87Dqzpw83oh93rKICdvHu0lSVolQTm+h5LtJknGrIFvApONqrMQ8M7RPzNdpojptC4PH6JA19fXZvCo9IjWCTwUOODrEy79WW23fmsvjLHpMgoydFsSbu3ots927esk7p3PrQadwr6RHvv1RWHDuWFRJVfU9khkanetMNESPDTfts11FfbtlPQFIaWnraS1FwqoYc2bC5Eyy5GlpIB/ddUkWS9LZuIjQgOPukzPDxNa82tXmmVJdGwjc4OhwBXIZOr1a2QC5ZG9cTzFMn55AaR34Rkcsde9TA2G0mYKODuMUp35FtVXz+G7E0IzT+qfXVI994iFE/LSuPKJkeOeAtcW//NsV0rS7slXYjoiVPEzx7UppOrpC3AwpKooClfffrpSm3RXZUehF7RTcIuG+X7VVB0wkpiFtCLCYZJx/X5gYTltL6JA2EuH3xImDWyzojHiqrg+1l98RKJRdomM4KbbS6rsCxVEMBO9cOx+8oP/8pKQ9w3rwsHcj4rlL1fL5Gm4e7RKyBu620frM3Z7nUWFFPP5wbSpesOxcI8o6/Hn9KH5SSNHdnrUyG+5fPR/jrSsIXMROHhA25uUZ1wpJpIh4eQ6+PQy8NYL7c/OQ5wnWxQUr2iFJ9OQ2+Gu0W08JTFSpmD4zVj6LJBsnSbZJ0i/RtN/HCUmdX6r1eomBfjWzE0lDythvRwgfxWs7hJbdR2UI2nZNZ2zDmgl7dhryDOHauCISbUWkZY3jjSyk9VRMSr8vYbz7cXh9KbHaXrbStdN2m2nIVmL1lTQkHSAb062v7+b42jQkk7E0ZPyHiXz7vN3utVmHoQa1SvZDnp2GPEew/IXscsv0U2lyrHbkqFyUSu1+JFi5jtOk2zQ1qDgc3I7Y6R9ancbOx5LRI6GNbKvT7mVIOjNBc+epCPLsY/nP9E7j5l+2zb+Lnl5OBwBL9CZo3VPbH4hQEz2YRIcTKhok2ZdcVcHVnrBztyTXD8RUqn2Wm3o9Ixk5JCuWkYz90gn9aFL7h966C7U1FaMAq56HSoj7L/c0rk+j8Iuc0/nDaDKlHzdk1dVyM36alkR2da2cLdYl8aIsVmFMd3SCD22r7VO6gX7ODO3Npk+hkbuLfyuW4UdVZ8/JpizyJC8+FrPF8qmYrxO0kk8DBeX7ZL5IpvMcT+MP7j1iAi0268FF/+L3hLvrKmmalM/z9WOxnmbX9DOiST5NH+bg7jTro4V1givFcrZ4rtoPmVVq4o+L5Df//ttksUpmRZoXq/EC7EimT8sVGBuenZZJNkunTxji+DlZP+J74F0CYU3DMP5SLNMVSEloblc3yyRd4YnkKE+bpPN09lxOy914VhhPmlWdrItyfQ2hFglhaVlQs78tpjNwPVktNvOcPjxMP6LhZbG6Dh3pX+36Wn9aJE8LdEgzD0NM5/hTJFVV3f659wn9GC2N62kB4JbJpymomSdFmj3uOXIP7JeLzSqjLPPP4X81Lv/pT78/Ym05SH6zWqXPSfiJ1zJZzGfPQWBAypqGDL4Wn4FHkJ4AakVZDqpGf56Dx4vVOgGOH/df5punJRhVJvPl/toynee4gv8v85/nPwMEk6Sa6aNA1mVJP1gLrk5z/FkuiN/ATT/ZqZOr25/nCf6rOv7XdPUAZifVWxjoqly/T8p1CvWV/OF3yXpaXI9XRfohsAdAA9fm9LOeRE71y7V7EqjVD8kQ1K8viY5BBpFd7vpNfgKm5pf7EV1dXVWvzBbQKqPFCnDDy/PlYFZ8LkHq5SVIuNs/f99PrqtRHl3btZE9LoAtvL6/dXfU7N3th/v77YMhS4IHJ7NFut4y6656/e6a399vGwTdI+i6LTUYVLl5umx1nwyH2xavjt7bD6b9anWr471/Sv5wNKmJfGLwlH4tmWZEGjhY/cZrAHPxGVwlgG+Fj9uDqqUFbhMjl/ngd5j5v1+lT8XlL73doHu3x6DohXHhWou0ftVa93+9aY536uL5cjWgYY8+pjNMwsvD63fbTvrhNYgxLbMiTOHh3e/TWVn0k7+uNsX91WDX2GC9GAXsX16R6KqmwEhSC3htkJZQqutRSnNtVPwNxvKyYmy/on7L01Wx3qzmye7WLz3SB73bOgqJB1tD2bv9gG+VXHq31b8RNvTqybje7RYvR60171bD+EJztpq1T0VabsD0y+c+pn35YTcx18stdJ7v6PJ9wM/VFRTJfm6VH/YXK0pIE25f2t6pseCY9Gci+cgL6d3um6jxYt4gvddwS/DETaO9w8vrZfPtDscELVzjwaS39xjoxZt5Mp2AVmiWIvnjYl40W6q8kPDogXi8cviyf/WI21tlT8yGtYDRKT7tGJ6CdcfCwH0wdty4Si/smBr8lvoEv3weDvlV8mt6Dn//HzWyV3BQsN0Pk+HCw9T29uHg2rSfZt1Nbz2eyPMdrYeZs6PguhrZMBnfNSV4f522r9XaCAO93g8gNNIl5NBS542qOUgODxyuDofj46+3B+nXB/8uDGA4JBrebYdRPRscu+ERQ4gP/zcIEHI9dgibD/06OeLVfu4gNOndpv3ekV/fux33e8dOMS7c7ZAZKN59bqK34SOH9w7fwqtHX5tv1xzn3u32Q793cJ8xyfG336t70PtHg8hfmJuBqaCW/un32r41NRQ+NNtoe9u92+Nv/d6R1927Df90t7F1wuuv34QXjuZzcDdHR54xzdHg80z6yTglq/I0/byb4s99ujXphxu4jglzB7mnZbAhl+kVnNUVFAH8v7vms/cN3M9ml+kA9usJc/il10gv/q/gGm/dwacNZtu4gFtYXON18oUXwROkMZSDXq0bUq0YIcgGV8gTuroC1nGRPsJ5zmH410dPPNP9X1iff6k1U39oQg/RpVU6fygudUMtEGmNVu8mw+GH+33bgd4PRO++BVD5L+Qyw+7TD86TnxL88UAtnMU14JeMYZeTNfmaawof0F2T3oqt5WO6LIbD5+rDEXebvG29izFPywkCs3UBcQ7oUsWmy8v0v0Id/voy/S/Qubs7h5abbW5/tX54d/8+BBBDwklFbWXuoEPIApfDX7asprb2Xi41Sb/cRqcDIWw/1l/0XBVYhKd6VUDS6/d2BS01bUf/7dQgunpPruWhz/0T2WKJy3eXPdI2fRrw1X1lDamj4XDXS2UU7y7L9eryw1V4cCvbtkzv633Q/Tm8yP42SKDBXwatGARy1b+sqUZi5lWTEvqPuEbc/HuxWpRbZvbHGB44mhfAyaxF3q77EGn0YeGp84rm227PbOvtgVnDrrCIcIRm+vtoqLsVGurOWR+Sd/q+GuBdGMj9EM23X9xJ646YdT+kNt6T0Lbfqxbqr5ErApMz3L8a2HrfP3w/5mxDLlv0o43K5xsO0VT1sf4ghD38pVeFrL3bHenAXcAIOYD0b79H1VS929724KoewWj38HAIQFcY6h1F8b0u//innw7O1pbAq37vqNIlOMRl16u9sG7du737JXwij/yotQqw1Ob2E1regrhDM91/6WQX8efzXdM+3of3P4f3F5/utuO4B0/D1+bTDf5WKmOQLimAu8QbDQE0GRmZHBROkSTe9UYBBkMM4n3jRg0PJPCas7IdSRVQlAdrma/ST5fVzX6yRGy+M4yHtMNytljPpuPq8uH7YFMWl73fPDz0riJvDJbP9CmkJGbr6iF8GKyyP0NvPpWDzRJ2gWLQCazBYJI+TWcAXe93xX+k/3uT/CWdl9CB4V45/TsAyEW/l36GhSiX0zn+WS+WvdsqUnwhKK29spo+PK53L/XK9GMxmT4M8uW0R8f4fdmp+tXwbhUEvwqCD/wh1y0wsRf06OpuN3Hug+i2okdzfepwSJSWmzFxoLzkfdHHHSJjeAky1MBe9R9WUDrLIht9oFn4aZqvH0crMpSEc97nA3G/G9DnI2MjoBXDw8OBlFtzhw7v2P1gnK4uP1+Hmzeif8cZ+2m11RwHx/OIrtV9PzzczxazxWrY+yctabWo1w/abNj773jzqqOLd80uauB7W1fcuonT+65C9ixx7EawpGqn0TH5H5/X0wz29XM/fKiyTMO73p+rBF7Q50BM5VvBdiaV7ew/7/qgiZpk6QYYSC5/ddXDHXhrl6xvDx7PoTuopPUMGP8LtHQifoUXl2kG3XabhBmfbKnszxbZEB7xZN3rE1g/FYSxYW9MnTdahd9MimBCuRdM+QqG9FJABmdXB8dh2g+MK+abp4Kypper1dVxzLNtcV18Xl9OaV5/vmzLuh+XzdVPkN07EZs5k58vfsHLtVDmnl65fTcQky/JcvnzRf8xHfaygrzx3hEV/JhoHiDzCFgOuO3fre4aoUoDIwMpYgj5TZ6H1Awc48mRIHudvb0LvV3vuwsR0At9ZWMrtN/39edV8XG62JToY509wrUJ7zd6Sj9/nEGpAD67VtJ0rMYZWvk0HLjG04Tf5x1+n0/C7+cGfktKZRDXKRV4+e5qm5b/GL5eX4U8YNVYeww7UP+fxxQe9yNplDz59FjMKW6nBv7bv/7Pf34Tpt8KUr4FaVv072R/GmBBvmf79lX/4wFfV5FWmyK+RpvvWm1W964CYCsT0NH2rl34OA9TxI2fhwNxtdfqUObLLe9+t6CFi+QRDUFItC4SYFKtLVTYTGnlIsy8hKBUfEoqYwHWolmmwkgO/K0mjonzGv0HculVRBG6DzAs/g6pHS8H0aLDUeCb/LxhbGwT433fG5asYT/KtH6Pqz5jbDvG4ERXyxyVp7dNjk1LRObHaqppL45HSYMfzdLnxQYih686vGMU7+g+Qh8v76/eB15WhveSPI7+JM2KbYOfHhGm9a7ek/HMIDREptOHq8pduaCVysV6gUdp9bRcp+tNeXG7W1UtgMdi9FSsV9NsROPeVMlxvJatwgraaLPO8Lxgwlxzfs3EX7m4FepW8oHknmv/DtqNsQuqqFyH7QNB0DSbKklialS2qVPe9FyZhqWpIPCdufhnNBgiE2pwvyRU5SmSBTmAQZCg56HgqrWmt10i6idHC4l9dIZpQkt8NxVckgXA+TT9exVnY2bSMuN2fpe0SlalGohxldoZIQQeTfNwhaSAwdF64E1YxBxk5Uca9WMqtMEdZ4s05Wzscm5TW6iJ0RZfUp4Z2ihG2zOkG+e5G5s0p52FXAueZixjhZ+Mc33xJazjjxaLyVF/tASarrLHEUmEcyZuyoIkmT7crIpJsSrmWXGDVwbz5fPxaMZUTJnSr2SLLOd8XGiVCq8cZ6mapEUxlqksJk6MWTGxeqLNZJLaVI3zMR1t5czFtgy2Qf6bhrN7rzkmrwxVaGo5MS7jBvxSE5+mTNo0zZly1vgc5kzlWYaBe+t5lsssZVZy2FYVOPQwe2pwqFrXudlC4waz8iOl3DC1IMwben7rknaxSeV2AkFwDAFzOtdGOZNzl6WZVmM5yTNrpFNCMpeCo452wY2NVTa3zE0g24t6xcLLQ8sAJ8rydArMcwuG5DZPLZ1+xiwrHARRTKQupBEmVd5PTOpS662EchM6LRje5RMjM2HAHAwlnU0f5qTmRqHromvPS1XbHOrfd2UV/PBRHD7Kw0e1+4g+9lRUi0flNqhyDJi4uF0j3P/S3x/ffLvbh9Q4QgfN7LXmQT0FtfE0ncOngFd9sd21FHZ33G3PdUkQz1x/CFo8WG88tJgX26t/2zsFkD7+NrJqF/e056laACfp0AJrNbqf5lfbBFPogzJfQcXTW+9pcZ2CtmoJB+p+Ax0fntnRBE/j85JGup49B81YjMKyLymzYrdeB6W1GKfj6Qzq7mgZL5keeghNU6oq6MrtamuVrawyGtW98Ngg+eNilxSsVsNDdqda+xwQLksYjHneLQXiKB1/3qfqGLBluVktYUkwYHh1FG9NJ9NQR1Fmq+mSFoaOyw+gc2FRV5QuwkvJmNbAqw5uQjkAqesn0vVlsQ/vaUSV4QnS3FMUiLnY1rDsCb04Lk05vnqoj9nX3Oze3a0sjZ8rA7S/ERZqjq4etZzOihUJkzL2N7sKmG1BS4Ivq1m6vPkf28KW/sW+XmI/fFpErsQGXBV7BG232x29sR/4TTXsINdlOl0FNiOAKAmelcYgBv17yCEnf/hdBbUFPLIpufXUYViXR+N/IHsF4zcrkse0JBf8Kdjcg5msXAFooE3+vNWD9B6tk08zChJoaNTHehEsd5VNp4KR3SyvFstnz0k12xM2cD9hcr5jA/ET3sCj/wI2PYdxgbIirCGSW0fJ5OSxmOXXVPhwsNhBcEcraNt258BxJS5KEhH+w439rNs6CqFEZrtm31zhP6rswXiz1aIsr7d6MPk4TcPkINlURSDkf6C130AUOvCh3E7DIu+jqWy2CdNzXjykAf+VkIK7UULLYoJAE2ZFVdATVEt4u9Itf4JOwuyDxKGb0g9F+il9ruSIGzvHMdSJ3MB526yKQfJncAkT5HGB7uHFrtLtlKrkj4l/XCuDOTdbfBokvyW2rpM/Tj9MMbgqdAue1a7GhqJoIoPaSLMP88UnwPOBBvf0tJlTKRFQVgb2PS3WZDqphCD5t+LlOqp9gVRCDtH7o6HBySpCgobAigm3LVFqFUz1MWHWiw3kkx/raRAQGEpOHwUNR9VT5KG9rTDsrYVmX778f7MwBW4=')))
recorded_queue_audit=s15_payload['record']
queue_audit_source_dir=output/'queue_audit_source'
queue_audit_source_dir.mkdir(exist_ok=True)
(queue_audit_source_dir/'audit_core.py').write_text(s15_payload['source'])
(queue_audit_source_dir/'protocol.json').write_text(json.dumps(s15_payload['protocol'],indent=2))
(output/'recorded_queue_audit.json').write_text(json.dumps(recorded_queue_audit,indent=2));

# %% cell 182
s15_primary=[r for r in recorded_queue_audit['results'] if r['budget']==.02]
s15_rows=[]
for r in s15_primary:
    s15_rows.append({'Queue':'Pooled OOF' if r['policy']=='pooled' else 'Separate folds',
        'Reviews':r['h10']['selected'],'H10 catches':r['h10']['fraud_selected'],
        'Blend catches':r['fixed_blend']['fraud_selected'],
        'Net catches':r['net_fraud_gain'],'Recall change (pp)':100*r['delta_recall']})
display(styled_table(pd.DataFrame(s15_rows).set_index('Queue'),{'Recall change (pp)':'{:+.3f}'}))

# %% cell 183
fig,ax=plt.subplots(figsize=(7.3,4.3),dpi=100)
s15_plot_rows=[r for r in recorded_queue_audit['results'] if r['budget']==.02]
s15_y=np.arange(2)
ax.barh(s15_y-.18,[r['fraud_rescued'] for r in s15_plot_rows],.32,color='#178f85',label='Additional fraud caught')
ax.barh(s15_y+.18,[-r['fraud_lost'] for r in s15_plot_rows],.32,color='#cb7259',label='Previous catches lost')
ax.axvline(0,color='#aab4bc',lw=.8)
ax.set_yticks(s15_y,['Pooled OOF','Separate folds']);ax.invert_yaxis()
ax.set_xlim(-65,65)
for i,r in enumerate(s15_plot_rows):
    ax.text(r['fraud_rescued']+2,i-.18,str(r['fraud_rescued']),va='center',fontsize=12)
    ax.text(-r['fraud_lost']-2,i+.18,str(r['fraud_lost']),ha='right',va='center',fontsize=12)
ax.set_xlabel('Fraud cases leaving (-) or entering (+) the review queue',fontsize=12)
ax.set_title('At 2% capacity, the blend loses more catches',loc='left',fontsize=14,pad=40)
ax.legend(frameon=False,fontsize=11,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2)
ax.tick_params(labelsize=12);ax.spines[['top','right']].set_visible(False);fig.tight_layout()
show_figure(fig,'Each policy reviews 14,000 of 699,960 transactions. The pooled blend catches 40 additional fraud cases but loses 49 previous catches; separate fold queues add 46 and lose 50. These are descriptive changes on reused development OOF predictions.','queue_audit.png')

# %% cell 186
s15_details=[]
for r in recorded_queue_audit['results']:
    s15_details.append({'Budget':r['budget'],'Queue':r['policy'],'Reviews':r['h10']['selected'],
        'H10 recall':r['h10']['recall'],'Blend recall':r['fixed_blend']['recall'],
        'H10 precision':r['h10']['precision'],'Blend precision':r['fixed_blend']['precision'],
        'Fraud rescued':r['fraud_rescued'],'Fraud lost':r['fraud_lost'],'Net catches':r['net_fraud_gain']})
s15_detail_table=pd.DataFrame(s15_details)
s15_detail_table.to_csv(output/'queue_budget_comparison.csv',index=False)
s15_fold_rows=[{'Queue':r['policy'],'Fold':f['fold'],'H10 reviews':f['h10']['selected'],
    'Blend reviews':f['fixed_blend']['selected'],'Net catches':f['net_fraud_gain'],
    'Recall change (pp)':100*f['delta_recall']} for r in s15_primary for f in r['folds']]
pd.DataFrame(s15_fold_rows).to_csv(output/'queue_primary_folds.csv',index=False)
display(styled_table(pd.DataFrame(s15_fold_rows).set_index(['Queue','Fold']),{'Recall change (pp)':'{:+.3f}'}))
display(styled_table(s15_detail_table.set_index(['Budget','Queue']),{
    'H10 recall':'{:.2%}','Blend recall':'{:.2%}','H10 precision':'{:.2%}','Blend precision':'{:.2%}'}))

# %% cell 188
QUEUE_AUDIT_INPUTS = None  # Set a local NPZ path to audit your aligned OOF arrays.
if QUEUE_AUDIT_INPUTS is not None:
    import importlib.util as s15_import
    s15_spec=s15_import.spec_from_file_location('s15_queue_audit_core',queue_audit_source_dir/'audit_core.py')
    s15_core=s15_import.module_from_spec(s15_spec)
    s15_spec.loader.exec_module(s15_core)
    with np.load(QUEUE_AUDIT_INPUTS,allow_pickle=False) as s15_data:
        s15_fresh_rows,_=s15_core.audit_predictions(s15_data['labels'],s15_data['ids'],s15_data['folds'],
            s15_data['h10_oof'],.8*s15_data['h10_oof']+.2*s15_data['glm_oof'])
    (output/'queue_audit_replay.json').write_text(json.dumps({'results':s15_fresh_rows,
        'scope':'User-supplied predictions; this replay does not establish training provenance.'},indent=2))
    print('Queue audit complete. The primary submission.csv is unchanged.')
else:
    print('Showing the recorded queue audit. Supply aligned OOF arrays to repeat it without fitting models.')
