import os
import pandas as pd
import time
from datetime import datetime
from decimal import Decimal

from joblib import Parallel, delayed

source = 'apple'
sleep_di = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'N4': 4, 'R': 5, 'L': 0}


def trans_to_timestamp(time_string):
    try:
        t = datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S.%f")
    except:
        time_string += '.000'
        t = datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S.%f")
    microsecond = t.microsecond / 1000000
    timeStamp = int(time.mktime(t.timetuple())) + microsecond
    return timeStamp


def trans_to_time(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")


def add_milliseconds(df):
    lens = df.shape[0]
    i, j = 0, 0
    value_count = dict(df.iloc[:, 0].value_counts())
    last_time = ''
    while i < lens:
        cur_time = df.iloc[i, 0]
        cur_count = value_count[cur_time]
        last_time_sep = 1000 // cur_count
        if cur_time != last_time:
            last_time = cur_time
            j = 0
        df.iloc[i, 0] = cur_time + Decimal('0.001') * j * last_time_sep
        i, j = i + 1, j + 1

    return df.iloc[:, 0]


def summarize_labeled(filenames):
    results = pd.DataFrame(columns=['sub_id', 'count', 'W', 'L', 'N1', 'N2', 'N3', 'R'])
    for files in filenames:
        if files.endswith('.csv'):
            file_start_time = time.time()
            sub_id = files.split('_')[0]
            print(sub_id)
            df = pd.read_csv(os.path.join(data_path, files))
            labels = df.loc[:, ['Apple ENMO', 'Apple Heart Rate', 'Time', 'Stg']].dropna()
            value_counts = labels['Stg'].value_counts()
            value_counts['sub_id'] = sub_id
            value_counts['count'] = labels.shape[0]
            results = pd.concat([results, pd.DataFrame(value_counts).T])
    results.fillna(0, inplace=True)
    # results.to_csv('./summarize_label_data_apple.csv', index=False)
    return results


def process_each_file(files):
    file_start_time = time.time()
    sub_id = files.split('.')[0]
    print(os.path.join(save_path, source, 'heart_rate', sub_id + '_heart_rate.csv'))
    if os.path.exists(os.path.join(save_path, source, 'heart_rate', sub_id + '_heart_rate.csv')):
        return
    df = pd.read_csv(os.path.join(data_path, files))

    upper_source = str.upper(source[0]) + source[1:]
    df.rename(columns={'PSG Time': 'psgtime', 'PSG Stg': 'psgstg'}, inplace=True)
    df.rename(columns={f'{upper_source} Time': f'{source}time', f'{upper_source} X': f'{source}x',
                       f'{upper_source} Y': f'{source}y', f'{upper_source} Z': f'{source}z',
                       f'{upper_source} Magnitude': f'{source}magnitude',
                       f'{upper_source} ENMO': f'{source}enmo',
                       f'{upper_source} Heart Rate': f'{source}heartrate',
                       f'Kubios Time': 'kubiostime', 'Kubios Medium Mean HR': 'kubiosmediummeanhr'},
              inplace=True)
    if all(column in df.columns for column in ['time', 'stg']):
        df.rename(columns={'time': 'psgtime', 'stg': 'psgstg'}, inplace=True)


    if source == 'actigraph':
        check_columns = [source + 'time', source + 'x', source + 'y', source + 'z',
                        source + 'magnitude', source + 'enmo', 'kubiostime', 
                        'kubiosmediummeanhr','psgtime', 'psgstg']
    else:
        check_columns = [source + 'time', source + 'x', source + 'y', source + 'z',
                        source + 'magnitude', source + 'enmo', source + 'heartrate'
                        ,'psgtime', 'psgstg']

    if not all(column in df.columns for column in check_columns):
        return

    source_data = df.loc[:, [source + 'time', source + 'x', source + 'y', source + 'z',
                             source + 'magnitude', source + 'enmo']].dropna()
    if len(source_data) == 0:
        return

    if source == 'actigraph':
        hr_data = df.loc[:, ['kubiostime', 'kubiosmediummeanhr']].dropna()
    else:
        hr_data = df.loc[:, [source + 'time', source + 'heartrate']].dropna()
    if len(hr_data) == 0:
        return

    labels = df.loc[:, ['psgtime', 'psgstg']].dropna()

    labels['labels'] = labels['psgstg'].apply(lambda x: sleep_di[x])

    labels['timestamp'] = labels['psgtime'].apply(
        lambda x: int(time.mktime(time.strptime(x if '.' not in x else x.split('.')[0], "%Y-%m-%d %H:%M:%S"))))
    start_time = labels['timestamp'].iloc[0]
    labels['Time'] = labels['timestamp'].apply(lambda x: x - start_time)
    os.makedirs(os.path.join(save_path, source, 'labels'), exist_ok=True)
    labels.to_csv(os.path.join(save_path, source, 'labels', sub_id + '_labeled.csv'), index=False)

    if '.' in source_data[source + 'time'].iloc[0]:
        source_data['timestamp'] = source_data[source + 'time'].apply(
            lambda x: trans_to_timestamp(x))
    else:
        source_data['timestamp'] = source_data[source + 'time'].apply(
            lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
        source_data['timestamp'] = add_milliseconds(pd.DataFrame(source_data['timestamp']))
    source_data[source + 'Time'] = source_data['timestamp'].apply(lambda x: x - start_time)
    if not os.path.exists(os.path.join(save_path, source, 'motion')):
        os.makedirs(os.path.join(save_path, source, 'motion'), exist_ok=True)
    source_data = source_data.loc[:,
                  [source + 'Time', source + 'x', source + 'y', source + 'z', source + 'magnitude', source + 'enmo',
                   'timestamp']]
    source_data.to_csv(os.path.join(save_path, source, 'motion', sub_id + '_motion.csv'), index=False)

    hr_time_column = source + 'time' if source != 'actigraph' else 'kubiostime'
    hr_heartrate_column = source + 'heartrate' if source != 'actigraph' else 'kubiosmediummeanhr'
    save_time_column = source + 'Time' if source != 'actigraph' else 'kubios' + 'Time'
    if '.' in hr_data[hr_time_column].iloc[0]:
        hr_data['timestamp'] = hr_data[hr_time_column].apply(lambda x: trans_to_timestamp(x))
    else:
        hr_data['timestamp'] = hr_data[hr_time_column].apply(
            lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
        hr_data['timestamp'] = add_milliseconds(pd.DataFrame(hr_data['timestamp']))

    hr_data[save_time_column] = hr_data['timestamp'].apply(lambda x: x - start_time)
    os.makedirs(os.path.join(save_path, source, 'heart_rate'), exist_ok=True)
    hr_data = hr_data.loc[:, [save_time_column, hr_heartrate_column, 'timestamp']]
    hr_data.to_csv(os.path.join(save_path, source, 'heart_rate', sub_id + '_heart_rate.csv'),
                   index=False)
    end_time = time.time()
    print(str(sub_id) + "  Execution took " + str((end_time - file_start_time) / 60) + " minutes")
    return


if __name__ == '__main__':
    data_path = '../data/raw_data'
    save_path = '../data/data_processed/'
    if source == 'actigraph':
        raw_filenames = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    else:
        ids = pd.read_csv(f'../data/{source}_ids.csv')['subject'].tolist()
        ids = [str(i) for i in ids]
        raw_filenames = [f for f in os.listdir(data_path) if f.split('.')[0] in ids]
    parallel = Parallel(n_jobs=os.cpu_count() - 2)
    print(f"Processing {len(raw_filenames)} files")
    all_results = parallel(delayed(process_each_file)(i) for i in raw_filenames)
