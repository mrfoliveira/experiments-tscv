import os

import pandas as pd

from utilsforecast.losses import mae
from modelradar.evaluate.radar import ModelRadar
from src.loaders import ChronosDataset, LongHorizonDatasetR

from src.cv import CV_METHODS
from src.mase import mase_scaling_factor
from src.utils import rename_uids
from src.config import OUT_SET_MULTIPLIER

RESULTS_DIR = "assets/results"

dataset_names = set(f.split(',')[0] for f in os.listdir(RESULTS_DIR))

MODELS = ["KAN", 'PatchTST', 'NBEATS', 'TFT',
          'TiDE', 'NLinear', "MLP",
          'DLinear', 'NHITS', 'DeepNPTS',
          "SeasonalNaive"]
FOLD_BASED_ERROR = False

cv_scores = []
for ds in dataset_names:
    print(ds)

    if ds in [*LongHorizonDatasetR.FREQUENCY_MAP]:
        df, horizon, _, _, seas_len = LongHorizonDatasetR.load_everything(ds)
    else:
        df, horizon, _, _, seas_len = ChronosDataset.load_everything(ds)

    in_set, _ = ChronosDataset.time_wise_split(df, horizon * OUT_SET_MULTIPLIER)
    mase_sf = mase_scaling_factor(seasonality=seas_len, train_df=in_set)

    cv_methods = [*CV_METHODS] + ['TimeHoldout']

    for method in cv_methods:
        # method = 'Holdout'

        inner_path = os.path.join(RESULTS_DIR, f"{ds},{method},inner.csv")
        outer_path = os.path.join(RESULTS_DIR, f"{ds},{method},outer.csv")

        if not os.path.isfile(inner_path) or not os.path.isfile(outer_path):
            continue

        cv_inner = pd.read_csv(inner_path)
        cv_inner.rename(columns={col: col.replace('Auto', '', 1)
                                 for col in cv_inner.columns if col.startswith('Auto')},
                        inplace=True)
        cv_outer = pd.read_csv(outer_path)

        radar_outer = ModelRadar(
            cv_df=cv_outer,
            metrics=[mae],
            model_names=MODELS,
            hardness_reference="SeasonalNaive",
            ratios_reference="SeasonalNaive",
        )

        # err_outer = radar_outer.evaluate(keep_uids=False)
        # err_outer /= mase_sf.mean()
        err_outer_uids = radar_outer.evaluate(keep_uids=True)
        err_outer = err_outer_uids.div(mase_sf, axis=0).mean()
        err_outer = err_outer.drop('SeasonalNaive')

        if FOLD_BASED_ERROR:
            cv_inner_g = cv_inner.groupby('fold')
            folds_res = []
            for g, fold_cv in cv_inner_g:
                fold_radar_inner = ModelRadar(
                    cv_df=fold_cv,
                    metrics=[mae],
                    model_names=MODELS,
                    hardness_reference="SeasonalNaive",
                    ratios_reference="SeasonalNaive",
                )

                f_err_inner_uids = fold_radar_inner.evaluate(keep_uids=True)
                f_err_inner_uids = rename_uids(f_err_inner_uids)
                f_err_inner = f_err_inner_uids.div(mase_sf, axis=0).mean()
                f_err_inner = f_err_inner.drop('SeasonalNaive')
                folds_res.append(f_err_inner)

            err_inner = pd.DataFrame(folds_res).mean()
        else:
            radar_inner = ModelRadar(
                cv_df=cv_inner,
                metrics=[mae],
                model_names=MODELS,
                hardness_reference="SeasonalNaive",
                ratios_reference="SeasonalNaive",
            )

            # err_inner = radar_inner.evaluate(keep_uids=False)
            # err_inner /= mase_sf.mean()
            err_inner_uids = radar_inner.evaluate(keep_uids=True)
            err_inner_uids = rename_uids(err_inner_uids)
            err_inner = err_inner_uids.div(mase_sf.loc[err_inner_uids.index], axis=0).mean()
            err_inner = err_inner.drop('SeasonalNaive')

        selected_model = err_inner.idxmin()
        best_model = err_outer.idxmin()

        mae_all = (err_inner - err_outer).abs().mean()
        me_all = (err_inner - err_outer).mean()
        perc_under = ((err_inner - err_outer) < 0).mean()
        # mean_sq_err = ((err_inner - err_outer) ** 2).mean()
        accuracy = int(selected_model == best_model)
        regret = err_outer[selected_model] - err_outer[best_model]
        mae_best = err_outer[best_model] - err_inner[best_model]
        mae_sele = err_outer[selected_model] - err_inner[selected_model]

        cv_scores.append(
            {
                'Dataset': ds,
                'Method': method,
                'MAE': mae_all,
                # 'me_all': me_all,
                # 'perc_under': perc_under,
                # 'mae_best': mae_best,
                # 'mae_sele': mae_sele,
                'Accuracy': accuracy,
                'Regret': regret,
            }
        )

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 30)

cv_df = pd.DataFrame(cv_scores)
cv_df.groupby('Method').mean(numeric_only=True)

cv_df_summ = cv_df.groupby('Method').mean(numeric_only=True).round(3)
print(cv_df_summ.to_latex(caption='---', label='tab:overall'))

print(cv_df_summ)

cv_df_summ = cv_df_summ.T

NAME_MAPPING = {
    'RepeatedBootstrap': 'K-Bootstrap',
    'RepeatedHoldout': 'K-Holdout',
    'MonteCarlo': 'MC-CV',
    'KFold': 'K-fold CV',
    'TimeHoldout': 'Time-CV',
}

cv_df_summ = cv_df_summ.rename(columns=NAME_MAPPING)


def to_latex_tab(df, round_to_n, transpose_on_iter:bool, rotate_cols: bool):
    if rotate_cols:
        # if transpose_on_iter:
        #     df.index = [f'\\rotatebox{{60}}{{{x}}}' for x in df.index]
        # else:
        #     df.columns = [f'\\rotatebox{{60}}{{{x}}}' for x in df.columns]

        df.columns = [f'\\rotatebox{{60}}{{{x}}}' for x in df.columns]


    annotated_res = []
    for i, r in df.round(round_to_n).iterrows():
        top_2 = r.sort_values().unique()[:2]
        if len(top_2) < 2:
            raise ValueError('only one score')

        best1 = r[r == top_2[0]].values[0]
        best2 = r[r == top_2[1]].values[0]

        r[r == top_2[0]] = f'\\textbf{{{best1}}}'
        r[r == top_2[1]] = f'\\underline{{{best2}}}'

        annotated_res.append(r)

    annotated_res = pd.DataFrame(annotated_res).astype(str)

    #if transpose_on_iter:
    #    annotated_res=annotated_res.T

    text_tab = annotated_res.to_latex(caption='CAPTION', label='tab:scores_by_ds')

    return text_tab

print(to_latex_tab(cv_df_summ, round_to_n=3, transpose_on_iter=False,rotate_cols=False))
print(to_latex_tab(cv_df_summ, round_to_n=3, transpose_on_iter=True,rotate_cols=False))