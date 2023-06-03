from pytorch_pretrained_bert import BertTokenizer

import logging
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)

import torch
import pandas as pd
import os
import random


def read_data(folderpath, max_files=500):
    files = os.listdir(folderpath)

    data = []
    for file in files[:max_files]:

        path = os.path.join(folderpath, file)
        with open(path, 'r') as f:
            content = "".join(f.readlines())
            data.append(content)

    return data


def tokenize(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ids = []
    for text in texts:
        ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))

    return ids


def partition(ids, max_len, total_len):
    """
    partition id in ids into blocks of max_len,
    remove last block to make sure every block is the same size
    """

    books = []
    for id in ids:
        book = torch.tensor([id[i:i+max_len] for i in range(0, len(id), max_len)][:-1], dtype=torch.int32)
        if book.size(0) > 30:
            books.append(book)

    for book in books:
        print(book.shape)

    return books


def create_bert_data(path="data/pg19/train", max_len=512, total_len=30, max_files=30):
    """
    :return: List[Tensor(length, max_len)], None
    """

    data = partition(tokenize(read_data(path, max_files=max_files)),
                     max_len=max_len,
                     total_len=total_len
                     )

    return data, None


#  EVERYTHING BELOW IS FOR BENCHMARKING


def read_context_file(filename):
    try:
        data = pd.read_json(filename, lines=True)
        data["Date"] = pd.to_datetime(data["Date"])
        data.set_index("Date", inplace=True)

    except Exception as e:
        print(f"ERROR: Filename {filename}")
        print(e)
        print(pd.read_json(filename, lines=True))

    return data


def read_context(tickers, mock_data=True):
    """
    :return: A dictionary with tickers as keys and dataframe of texts as value

    TODO: Find out why split_ids sometime create NaNs
    """
    context = dict()
    srcdir = "data"
    for ticker in tickers:
        news = []
        news_path = f"{srcdir}/news/{ticker}"
        if os.path.exists(news_path):
            years = os.listdir(news_path)
            for year in years:
                news.append(read_context_file(f"{news_path}/{year}"))
        if len(news) != 0: news = pd.concat(news)
        else: news = pd.DataFrame(columns=["Date", "Url", "Text", "Ids"])

        tweets = []
        tweets_path = f"{srcdir}/tweets/{ticker}"
        if os.path.exists(tweets_path):
            years = os.listdir(tweets_path)
            for year in years:
                tweets.append(read_context_file(f"{tweets_path}/{year}"))
        if len(tweets) != 0: tweets = pd.concat(tweets)
        else: tweets = pd.DataFrame(columns=["Date", "Url", "Text", "Ids"])

        # final processing
        c = pd.concat([news, tweets]).sort_index()
        c = c.drop(columns=["Date", "Url", "Text"], axis=1)
        c = c.dropna()

        try:
            c = split_ids(c, maxlen=250)
        except Exception as e:
            print(f"ERROR: Ticker {ticker}")
            print(e)
            print(c)

        # split ids sometimes generate nans
        c = c.dropna()

        # temporary
        if mock_data:
            c = c[~c.index.duplicated(keep='first')]
            c.index = c.index.date
            c = c[~c.index.duplicated(keep='first')]
            c.index = pd.to_datetime(c.index)
            c = c.iloc[::5, :]

        assert not c.isnull().values.any()
        context[ticker] = c

    return context


def split_ids(dataframe, maxlen=250):
    """
    :return: Create new rows for strings with word count exceeding maxlen
    """
    def split(ids):
        lst = [ids[i:i+maxlen] for i in range(0, len(ids), maxlen)]
        return lst

    if len(dataframe.index) > 0:
        dataframe["Ids"] = dataframe.apply(lambda row : split(row["Ids"]), axis=1)
        dataframe = dataframe.explode(["Ids"])

    return dataframe


def read_prices(tickers, start, end):
    prices = []
    for ticker in tickers:
        price = pd.read_csv(f"data/prices/{ticker}.csv")
        price["Date"] = pd.to_datetime(price["Date"])
        price.set_index("Date", inplace=True)

        prices.append(price)

    prices = pd.concat(prices, axis=1, keys=tickers)
    prices = prices.iloc[:, prices.columns.get_level_values(1) == "Close"]
    prices = prices.loc[start:end]

    return prices


def create_stocks_data(mock_data=True):
    """
    [1 0 0 0 1 0 0 0 1 0 0 0 ...]
    """
    context = read_context(["AAPL", "AMZN"], mock_data=mock_data)

    prices = read_prices(tickers=["AAPL", "AMZN"],
                         start=pd.Timestamp("2018-01-01"), # 2020-01-01
                         end=pd.Timestamp("2021-01-01"))
    index = prices.index.tolist()

    X = []
    for idx in index:
        start = (idx - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        end = (idx - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        timeslice1 = context["AAPL"][start:end]["Ids"]
        timeslice2 = context["AMZN"][start:end]["Ids"]

        sample1 = timeslice1.sample(n=min(len(timeslice1), 1))
        sample2 = timeslice2.sample(n=min(len(timeslice2), 1))

        ids = []

        if not sample1.empty: ids += sample1.sum()
        if len(ids) < 250   : ids += ([102] * (250-len(ids)))

        ids += [102]

        if not sample2.empty: ids += sample2.sum()
        if len(ids) < 501   : ids += ([102] * (501-len(ids)))

        X.append(torch.tensor(ids, dtype=torch.int32))

    prices = prices.reset_index(drop=True)
    prices = prices["AAPL"] / prices["AMZN"]
    prices = prices["Close"].tolist()

    X = [torch.stack(X)]
    Y = [torch.tensor(prices, dtype=torch.float32).unsqueeze(1)]

    if mock_data:
        Y = [torch.rand(*Y[0].size())]

    print(X)
    print(Y)

    return X, Y


def create_memory_data():
    """
    [2 3 4 5 6]
    [1 2 3 4 5]
    """
    sample = torch.rand(1001, 1)
    X = sample[1:].repeat(1, 501)
    Y = sample[:-1]

    X = [X]
    Y = [Y]

    print(X)
    print(Y)

    return X, Y


def create_binary_data():
    """
    Create one hot X and Y with a 1 in X corresponding with a high reward in 20 timesteps and 50 timesteps
    (Requires memory within unroll length and memory outside unroll length)
    """
    X = []
    Y = []

    for _ in range(1):

        temp_X = torch.zeros(2000, 501, dtype=torch.int32)
        temp_Y = torch.zeros(2000, 1, dtype=torch.float32)

        for i in range(temp_X.size(0)-60):
            if random.random() < 0.05:
                temp_X[i][:250] = torch.ones(250,)
                for j in range(5, 60, 5):
                    temp_Y[i + j] = torch.tensor([1])

            if random.random() < 0.05:
                temp_X[i][251:] = torch.ones(250,)
                for j in range(40, 60):
                    temp_Y[i + j] = torch.tensor([1])

        X.append(temp_X)
        Y.append(temp_Y)

    print(X)
    print(Y)

    return X, Y


if __name__ == "__main__":
    X, Y = create_stocks_data()

    print(X)
    print(Y)

    print(X[0].shape)
    print(Y[0].shape)
