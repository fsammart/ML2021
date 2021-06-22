import pandas as pd
from sklearn.preprocessing import StandardScaler


# balances dataframe where column can take value 0 or 1
def balance_dataframe(df, column):
    df_zeros = df[df[column] == 0]
    df_ones = df[df[column] == 1]

    zeros_qty = len(df_zeros)
    ones_qty = len(df_ones)

    if zeros_qty > ones_qty:
        selected = df_zeros.head(ones_qty)
        return pd.concat([selected, df_ones], axis=0)
    else:
        selected = df_ones.head(zeros_qty)
        return pd.concat([selected, df_zeros], axis=0)


# returns pandas df
# attributes: list of attributes to remove
# scaler: sklearn scaler to fit and transform data
def load_data(filename, attributes=None, balance=None):
    df = pd.read_csv(filename, delimiter=';')
    if balance:
        df = balance_dataframe(df, balance)
    df_info = get_dataset_info(df)
    df.sample(frac=1, random_state=42).reset_index(drop=True)  # reproducible results
    if attributes:
        df = df[attributes]
    return df, df_info


# inplace replaces nan with most frequent value for each column
def replace_nan(df: pd.DataFrame, differentiate_label=False):
    nan_counts = []
    columns = df.columns
    for col in columns:
        nan_counts.append((col, df[col].isna().sum()))
        frequent = df[col].mode()[0]
        df[col] = df[col].fillna(float(frequent))
    return nan_counts


def scale_data(df):
    # TODO: justify why this one
    scaler = StandardScaler()
    headers = df.columns
    np_df = df.values
    scaled = scaler.fit_transform(np_df)
    return scaler, pd.DataFrame(scaled, columns=headers)


# customized for this dataset specifically
def get_dataset_info(df):
    total_registers = len(df)
    info = f'Dataframe has {total_registers} registers.\n'

    df_zeros = df[df['sigdz'] == 0]
    df_ones = df[df['sigdz'] == 1]
    zeros_qty = len(df_zeros)
    ones_qty = len(df_ones)
    info += f'{zeros_qty} registers have value 0 and {ones_qty} registers have value 1\n'

    nan_df = df[df['choleste'].isna()]
    nan_without_disease = nan_df[nan_df['sigdz'] == 0]
    nan_with_disease = nan_df[nan_df['sigdz'] == 1]
    info += f'Out of {len(nan_df)} registers with cholesterol nan, {len(nan_with_disease)} sigdz=1, {len(nan_without_disease)} sigdz=0\n'

    nan_choleste_qty = df['choleste'].isna().sum()
    info += f'Out of {total_registers}, {nan_choleste_qty} are nan in choleste column ({round(nan_choleste_qty/total_registers, 4)})\n'

    choleste_frequent = float(df['choleste'].mode())
    choleste_avg = float(df['choleste'].mean())
    choleste_std = float(df['choleste'].std())
    info += f'For choleste, mode is {choleste_frequent}, mean is {round(choleste_avg, 4)} and std is {round(choleste_std, 4)}\n'

    with_label = ['sex', 'age', 'cad.dur', 'choleste', 'sigdz']
    all_attributes = ['sex', 'age', 'cad.dur', 'choleste']
    numeric_attributes = ['age', 'cad.dur', 'choleste']

    with_label_duplicate = len(df[with_label])-len(df[with_label].drop_duplicates(inplace=False))
    info += f'With label duplicated\n{with_label_duplicate}\n'

    all_attributes_duplicate = len(df[all_attributes])-len(df[all_attributes].drop_duplicates(inplace=False))
    info += f'All attributes duplicated\n{all_attributes_duplicate}\n'

    numeric_attributes_duplicate = len(df[numeric_attributes])-len(df[numeric_attributes].drop_duplicates(inplace=False))
    info += f'Numeric attributes duplicated\n{numeric_attributes_duplicate}\n'

    return info
