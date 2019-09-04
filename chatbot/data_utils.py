import random
import numpy as np
from tensorflow.python.client import device_lib
from word_sequence import WordSequence

VOCAB_SIZE_THRESHOLD_CPU = 50000

'''获取当前GPU信息'''
def _get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

'''根据输入输出的字典大小来选择，是在CPU上embedding还是在GPU上进行embedding'''
def _get_embed_device(vocab_size):
    gpus = _get_available_gpus()
    if not gpus or vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        return "/cpu:0"
    return "/gpu:0"


def transform_sentence(sentence, ws, max_len=None, add_end=False):
    """ 单独的句子的转换 """
    encoded = ws.transform(
        sentence,
        max_len= max_len if max_len is not None else len(sentence))
    encoded_len = len(sentence) + (1 if add_end else 0)
    if encoded_len > len(encoded):
        encoded_len = len(encoded)
    #[4, 4, 5, 6]
    return encoded, encoded_len


def batch_flow(data, ws, batch_size, raw=False, add_end=True):
    """
    从数据中随机去生成batch_size的数据，然后给转换后输出出去
    raw:是否返回原始对象，如果为True，假设结果ret， 那么len(ret) == len(data) * 3
        如果为false，那么len(ret) == len(data） * 2
    Q = (q1, q2, q3 ... qn)
    A = (a1, a2, a3 ... an)
    len(Q) == len(A)
    batch_flow([Q, A], ws, batch_size = 32)
    raw = False:
    next(generator) == q_i_encoded, q_i_len, a_i_encoded, a_i_len
    raw = True:
    next(generator) == q_i_encoded, q_i_len, q_i, a_i_encoded, a_i_len, a_i
    """
    #ws数量要和data数量要保持一致（多个）,len(data) == len(ws)
    all_data = list(zip(*data))
    if isinstance(ws, (list, tuple)):
        assert len(ws) == len(data), 'ws的长度必须等于data的长度 if ws 是一个list or tuple'

    if isinstance(add_end, bool):
        add_end = [add_end] * len(data)
    else:
        assert(isinstance(add_end, (list, tuple))), 'add_end不是boolean，纠结应该是一个list(tuple) of boolean'
        assert len(add_end) == len(data), '如果add_end 是list(tuple)，那么add_end的长度应该和输入数据的长度一致'

    mul = 2
    if raw:
        mul = 3

    while True:
        data_batch = random.sample(all_data, batch_size) #在all_data数据中随机抽取生成batch_size个数据
        batches = [[] for i in range(len(data) * mul)]

        max_lens = []
        for j in range(len(data)):
            max_len = max([
                len(x[j]) if hasattr(x[j], '__len__') else 0
                for x in data_batch
            ]) + (1 if add_end[j] else 0)
            max_lens.append(max_len)

        for d in data_batch:
            for j in range(len(data)):
                if isinstance(ws, (list, tuple)):
                    w = ws[j]
                else:
                    w = ws

                #添加结束标记（结尾）
                line = d[j]
                if add_end[j] and isinstance(line, (tuple, list)):
                    line = list(line) + [WordSequence.END_TAG]
                if w is not None:
                    x, xl = transform_sentence(line, w, max_lens[j], add_end[j])
                    batches[j * mul].append(x)
                    batches[j * mul + 1].append(xl)
                else:
                    batches[j * mul].append(line)
                    batches[j * mul + 1].append(line)
                if raw:
                    batches[j * mul + 2].append(line)
        batches = [np.asarray(x) for x in batches]
        yield batches

def batch_flow_bucket(data, ws, batch_size, raw=False, add_end=True,
                      n_bucket=5, bucket_ind=1, debug=False):
    #bucket_ind 是指哪一个维度的输入作为bucket的依据
    #n_bucket就是指把数据分成了多少个bucket
    all_data = list(zip(*data))
    lengths = sorted(list(set([len(x[bucket_ind]) for x in all_data])))
    if n_bucket > len(lengths):
        n_bucket = len(lengths)

    splits = np.array(lengths)[
        (np.linspace(0, 1, 5, endpoint=False) * len(lengths)).astype(int)
    ].tolist()

    splits += [np.inf] #np.inf无限大的正整数

    if debug:
        print(splits)

    ind_data = {}
    for x in all_data:
        l = len(x[bucket_ind])
        for ind, s in enumerate(splits[:-1]):
            if l >= s and l <= splits[ind + 1]:
                if ind not in ind_data:
                    ind_data[ind] = []
                ind_data[ind].append(x)
                break

    inds = sorted(list(ind_data.keys()))
    ind_p = [len(ind_data[x]) / len(all_data) for x in inds]
    if debug:
        print(np.sum(ind_p), ind_p)

    if isinstance(ws, (list, tuple)):
        assert len(ws) == len(data), "len(ws) 必须等于len(data)，ws是list或者是tuple"

    if isinstance(add_end, bool):
        add_end = [add_end] * len(data)
    else:
        assert(isinstance(add_end, (list, tuple))), "add_end 不是 boolean，就应该是一个list(tuple) of boolean"
        assert len(add_end) == len(data), "如果add_end 是list(tuple)，那么add_end的长度应该和输入数据长度是一致的"

    mul = 2
    if raw:
        mul = 3

    while True:
        choice_ind = np.random.choice(inds, p=ind_p)
        if debug:
            print('choice_ind', choice_ind)
        data_batch = random.sample(ind_data[choice_ind], batch_size)
        batches = [[] for i in range(len(data) * mul)]

        max_lens = []
        for j in range(len(data)):
            max_len = max([
                len(x[j]) if hasattr(x[j], '__len__') else 0
                for x in data_batch
            ]) + (1 if add_end[j] else 0)

            max_lens.append(max_len)

        for d in data_batch:
            for j in range(len(data)):
                if isinstance(ws, (list, tuple)):
                    w = ws[j]
                else:
                    w = ws

                #添加结尾
                line = d[j]
                if add_end[j] and isinstance(line, (tuple, list)):
                    line = list(line) + [WordSequence.END_TAG]

                if w is not None:
                    x, xl = transform_sentence(line, w, max_lens[j], add_end[j])
                    batches[j * mul].append(x)
                    batches[j * mul + 1].append(xl)
                else:
                    batches[j * mul].append(line)
                    batches[j * mul + 1].append(line)
                if raw:
                    batches[j * mul + 2].append(line)
        batches = [np.asarray(x) for x in batches]

        yield batches

def test_batch_flow():
    from fake_data import generate
    x_data, y_data, ws_input, ws_target = generate(size=10000)
    flow = batch_flow([x_data, y_data], [ws_input, ws_target], 4)
    x, xl, y, yl = next(flow)
    print(x.shape, y.shape, xl.shape, yl.shape)

def test_batch_flow_bucket():
    from fake_data import generate
    x_data, y_data, ws_input, ws_target = generate(size=10000)
    flow = batch_flow_bucket([x_data, y_data], [ws_input, ws_target], 4, debug=True)
    for _ in range(10):
        x, xl, y, yl = next(flow)
        print(x.shape, y.shape, xl.shape, yl.shape)


if __name__ == '__main__':
    # size = 300000
    # print(_get_embe1d_device(size))
    test_batch_flow()
    # test_batch_flow_bucket()