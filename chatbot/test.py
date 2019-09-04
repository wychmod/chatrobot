import sys
import pickle

import numpy as np
import tensorflow as tf
from flask import Flask,request

def test(params, infos):

    from sequence_to_sequence import SequenceToSequence
    from data_utils import batch_flow

    x_data, _ = pickle.load(open('chatbot.pkl', 'rb'))
    ws = pickle.load(open('ws.pkl', 'rb'))

    for x in x_data[:5]:
        print(' '.join(x))

    config = tf.ConfigProto(
        device_count = {'CPU':1, 'GPU':0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = './model/s2ss_chatbot.ckpt'

    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=0,
        **params
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        while True:
            # user_text = input('请输入您的句子:')
            # if user_text in ('exit', 'quit'):
            #     exit(0)
            x_test = [list(infos.lower())]
            bar = batch_flow([x_test], ws, 1)
            x, xl = next(bar)
            x = np.flip(x, axis=1)

            print(x, xl)

            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print(pred)

            print(ws.inverse_transform(x[0]))

            for p in pred:
                ans = ws.inverse_transform(p)
                print(ans)
                return ans

app = Flask(__name__)

@app.route('/api/chatbot', methods=['get'])
def chatbot():
    infos = request.args['infos']

    import json
    text = test(json.load(open('params.json')), infos)
    # return text
    return "".join(text)

if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0', port=8000)

