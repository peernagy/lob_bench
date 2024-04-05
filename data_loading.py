import pandas as pd
from decimal import Decimal
from typing import Optional
import glob


def load_message_df(m_f: str) -> pd.DataFrame:
    cols = ['time', 'event_type', 'order_id', 'size', 'price', 'direction']
    messages = pd.read_csv(
        m_f,
        names=cols,
        usecols=cols,
        index_col=False,
        dtype={
            #'time': 'float64',
            'time': str,
            'event_type': 'int32',
            'order_id': 'int32',
            'size': 'int32',
            'price': 'int32',
            'direction': 'int32'
        }
    )
    messages.time = messages.time.apply(lambda x: Decimal(x))
    return messages

def load_book_df(b_f: str) -> pd.DataFrame:
    book = pd.read_csv(
        b_f,
        index_col=False,
        header=None
    )
    return book

def add_date_to_time(df: pd.DataFrame, date: str) -> pd.Series:
    # df.time = pd.to_datetime(date) + pd.to_datetime(df.time, unit='s')
    sec_nanosec = df.time.astype(str).str.split('.', expand=True)
    sec_nanosec[1] = sec_nanosec[1].str.pad(9, side='right', fillchar='0')
    df.time = pd.to_datetime(date) + pd.to_timedelta(sec_nanosec[0] + 's') + pd.to_timedelta(sec_nanosec[1] + 'ns')
    return df

class Simple_Loader():
    def __init__(self, real_data_path: str, gen_data_path: str) -> None:
        # load sample lobster data
        real_message_paths = glob.glob(real_data_path + '/*message*.csv')
        real_book_paths = glob.glob(real_data_path + '/*orderbook*.csv')
        if len(real_book_paths) == 0:
            real_book_paths = [None] * len(real_message_paths)

        self.paths = []
        for rmp, rbp in zip(real_message_paths, real_book_paths):
            _, _, after = rmp.partition('real_id_')
            real_id = after.split('_')[0]
            gen_messsage_paths = glob.glob(gen_data_path + f'/*message*real_id_{real_id}*gen_id*.csv')
            gen_book_paths = glob.glob(gen_data_path + f'/*orderbook*real_id_{real_id}*gen_id*.csv')
            date_str = rmp.split('/')[-1].split('_')[1]
            self.paths.append((date_str, rmp, rbp, gen_messsage_paths, gen_book_paths))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[pd.DataFrame, pd.DataFrame, tuple[pd.DataFrame], Optional[tuple[pd.DataFrame]]]:
        """ Get 1 real and N generated dataframes for a given period
            Returns: real_messages, real_book, tuple(gen_messages), [tuple(gen_books)]
            The generated book files are optional and will be calculated from messages using JaxLob simulator if not provided.
        """

        date, rmp, rbp, gmp, gbp = self.paths[i]

        # print(self.paths[i])
        m_real = load_message_df(rmp)
        m_real = add_date_to_time(m_real, date)
        b_real = load_book_df(rbp)
        m_gen = tuple(
            add_date_to_time(
                load_message_df(m), 
                date
            ) for m in gmp)
        b_gen = tuple(load_book_df(b) for b in gbp)

        return m_real, b_real, m_gen, b_gen
