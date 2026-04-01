import pandas as pd

METHOD_NAME_MAPPING = {
    'RepeatedBootstrap': 'K-Bootstrap',
    'RepeatedHoldout': 'K-Holdout',
    'MonteCarlo': 'MC-CV',
    'KFold': 'K-fold CV',
    'TimeHoldout': 'Time-CV',
}

DATA_NAME_MAPPING = {
    'monash_hospital': 'Hospital',
    'monash_m1_monthly': 'M1-M',
    'monash_m1_quarterly': 'M1-Q',
    'monash_m3_monthly': 'M3-M',
    'monash_m3_quarterly': 'M3-Q',
    'monash_tourism_monthly': 'Tourism-M',
    'monash_tourism_quarterly': 'Tourism-Q',
}


def rename_uids(df: pd.DataFrame) -> pd.DataFrame:
    if "fold" in df.index[0]:
        base_uid_list = df.index.str.split('_').map(
            lambda x: '_'.join(x[:-2]) if len(x) > 2 else df.index[0])
        df_cln = df.copy()
        df_cln.index = base_uid_list
        df_cln = df_cln.groupby(level=0).mean()
    else:
        df_cln = df

    return df_cln


def to_latex_tab(df, round_to_n, rotate_cols: bool):
    if rotate_cols:
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

    text_tab = annotated_res.to_latex(caption='CAPTION', label='LABEL')

    return text_tab
