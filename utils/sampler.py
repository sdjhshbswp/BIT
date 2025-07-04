import numpy as np
import pandas as pd

from utils import const


class Sampler(object):
    def __init__(self, data_path, search, user_vocab) -> None:
        self.search = search
        self.user_vocab = user_vocab
        self.data = pd.read_pickle(data_path)

    def sample(self, index):
        feed_dict = {}
        line = self.data.iloc[index]

        user = int(line['user_id'])
        feed_dict['user'] = [user]

        feed_dict['item'] = [int(line['item_id'])]

        feed_dict['neg_items'] = [line['neg_items']]

        feed_dict['search'] = self.search
        if self.search:
            query = self.get_pad_query(line['keyword'])
            feed_dict['query'] = list([query])

        rec_his_num = int(line['rec_his'])
        src_session_his_num = int(line['src_session_his'])
        feed_dict.update(
            self.get_all_his(user, rec_his_num, src_session_his_num))

        return feed_dict

    def get_all_his(self, user, rec_his_num, src_his_num):
        rec_his_item = self.user_vocab[user]['rec_his'][:rec_his_num][
            -const.max_rec_his_len:]
        rec_his_ts = self.user_vocab[user]['rec_his_ts'][:rec_his_num][
            -const.max_rec_his_len:]
        if len(rec_his_item) < const.max_rec_his_len:
            rec_his_item += [0] * (const.max_rec_his_len - len(rec_his_item))
            rec_his_ts += [np.inf] * (const.max_rec_his_len - len(rec_his_ts))
        rec_his_type = [1] * len(rec_his_item)
        rec_his = list(zip(rec_his_item, rec_his_ts, rec_his_type))

        src_his_item = self.user_vocab[user]['src_session_his'][:src_his_num][
            -const.max_src_session_his_len:]
        src_his_ts = self.user_vocab[user]['src_session_his_ts'][:src_his_num][
            -const.max_src_session_his_len:]
        if len(src_his_item) < const.max_src_session_his_len:
            src_his_item += [0] * \
                (const.max_src_session_his_len - len(src_his_item))
            src_his_ts += [np.inf] * \
                (const.max_src_session_his_len - len(src_his_ts))
        src_his_type = [2] * len(src_his_item)
        src_his = list(zip(src_his_item, src_his_ts, src_his_type))

        all_his = rec_his + src_his

        sorted_all_his = sorted(all_his, key=lambda x: x[1])
        sorted_all_his_item = [x[0] for x in sorted_all_his]
        sorted_all_his_time = [x[1] for x in sorted_all_his]
        sorted_all_his_type = [x[2] for x in sorted_all_his]

        return {
            "all_his": [sorted_all_his_item],
            "all_his_ts": [sorted_all_his_time],
            "all_his_type": [sorted_all_his_type]
        }

    def get_pad_query(self, query):
        if type(query) == str:
            query = eval(query)
        if type(query) == int:
            query = [query]
        query = query[:const.max_query_word_len]
        if len(query) < const.max_query_word_len:
            query += [0] * (const.max_query_word_len - len(query))
        return query


class WeixinSampler(object):
    def __init__(self, data_path, search, user_vocab) -> None:
        self.search = search
        self.user_vocab = user_vocab
        self.data = pd.read_pickle(data_path)
        self.data = self.data[self.data['1200_remap'].apply(lambda x: len(x) > 0)]


    def sample(self, index):
        feed_dict = {}
        line = self.data.iloc[index]

        user = int(line['new_userid'])
        _9000 = int(line['new_9000'])
        _9001 = int(line['new_9001'])
        _9002 = int(line['new_9002'])
        _9003 = int(line['new_9003'])
        _9004 = int(line['new_9004'])
        if "query_id" in line.keys():
            query_id = int(line['query_id'])
            feed_dict['query_id'] = [query_id]
        feed_dict['user'] = [user]
        feed_dict['_9000'] = [_9000]
        feed_dict['_9001'] = [_9001]
        feed_dict['_9002'] = [_9002]
        feed_dict['_9003'] = [_9003]
        feed_dict['_9004'] = [_9004]
        feed_dict['user'] = [user]
        feed_dict['item'] = [int(line['203_remap'][0])]
        feed_dict['title'] = [self.get_pad_query(line['200_remap'])]
        feed_dict['neg_items'] = [[int(eval(x)[0] if isinstance(eval(x), list) else eval(x)) for x in line['203_neg']]]

        feed_dict['search'] = self.search
        if self.search:
            query = self.get_pad_query(line['102_remap'])
            feed_dict['query'] = list([query])

        # rec_his_num = int(line['rec_his'])
        # src_session_his_num = int(line['src_session_his'])
        feed_dict.update(
            self.get_all_his(line))

        return feed_dict

    def get_all_his(self, line):

        rec_his_item = line["1203_remap"][
            -const.max_rec_his_len:]
        if len(rec_his_item) < const.max_rec_his_len:
            rec_his_item += [0] * (const.max_rec_his_len - len(rec_his_item))
        rec_his_item_1200 = line["1200_remap"][
            -const.max_rec_his_len:]
        if len(rec_his_item_1200) < const.max_rec_his_len:
            rec_his_item_1200 += [0] * (const.max_rec_his_len - len(rec_his_item_1200))        
        # rec_his_type = [1] * len(rec_his_item)
        # rec_his = list(zip(rec_his_item, rec_his_ts, rec_his_type))

        # src_his_item = self.user_vocab[user]['src_session_his'][:src_his_num][
        #     -const.max_src_session_his_len:]
        # src_his_ts = self.user_vocab[user]['src_session_his_ts'][:src_his_num][
        #     -const.max_src_session_his_len:]
        # if len(src_his_item) < const.max_src_session_his_len:
        #     src_his_item += [0] * \
        #         (const.max_src_session_his_len - len(src_his_item))
        #     src_his_ts += [np.inf] * \
        #         (const.max_src_session_his_len - len(src_his_ts))
        # src_his_type = [2] * len(src_his_item)
        # src_his = list(zip(src_his_item, src_his_ts, src_his_type))

        # all_his = rec_his + src_his

        # sorted_all_his = sorted(all_his, key=lambda x: x[1])
        # sorted_all_his_item = [x[0] for x in sorted_all_his]
        # sorted_all_his_time = [x[1] for x in sorted_all_his]
        # sorted_all_his_type = [x[2] for x in sorted_all_his]

        return {
            "all_his_1203": [rec_his_item],
            "all_his_1200": [rec_his_item_1200],
        }

    def get_pad_query(self, query):
        if type(query) == str:
            query = eval(query)
        if type(query) == int:
            query = [query]
        query = query[:const.max_query_word_len]
        if len(query) < const.max_query_word_len:
            query += [0] * (const.max_query_word_len - len(query))
        return query


