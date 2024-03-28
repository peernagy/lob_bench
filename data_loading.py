import pandas as pd
from decimal import Decimal


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