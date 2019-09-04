import random
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def test(params):

    from sequence_to_sequence import SequenceToSequence
    from data_utils import batch_flow_bucket as batch_flow
    from word_sequence import WordSequence
    from threadedgenerator import ThreadedGenerator

    x_data, y_data = pickle.load(open('chatbot.pkl', 'rb'))
    ws = pickle.load(open('ws.pkl', 'rb'))

    n_epoch = 2
    batch_size = 128
    steps = int(len(x_data) / batch_size) + 1

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = 'model/s2ss_chatbot_anti.ckpt'

    tf.reset_default_graph()
    with tf.Graph().as_default():
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        with tf.Session(config=config) as sess:
            model = SequenceToSequence(
                input_vocab_size=len(ws),
                target_vocab_size=len(ws),
                batch_size=batch_size,
                **params
            )
            init = tf.global_variables_initializer()
            sess.run(init)

            flow = ThreadedGenerator(
                batch_flow([x_data, y_data], ws, batch_size,add_end=[False, True]),
                queue_maxsize=30
            )

            dummy_encoder_inputs = np.array([
                np.array([WordSequence.PAD]) for _ in range(batch_size)])
            dummy_encoder_inputs_length = np.array([1] * batch_size)

            for epoch in range(1, n_epoch+1):
                costs = []
                bar = tqdm(range(steps), total=steps,
                           desc='epoch {}, loss=0.000000'.format(epoch))
                for _ in bar:
                    x, xl, y, yl = next(flow)
                    x = np.flip(x, axis=1)

                    add_loss = model.train(sess,
                                           dummy_encoder_inputs,
                                           dummy_encoder_inputs_length,
                                           y, yl, loss_only=True)
                    add_loss *= -0.5

                    cost, lr = model.train(sess, x, xl, y, yl,
                                           return_lr=True, add_loss=add_loss)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f} lr={:.6f}'.format(
                        epoch,
                        np.mean(costs),
                        lr
                    ))
                model.save(sess, save_path)
            flow.close()

    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=12,
        **params
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        bar = batch_flow([x_data, y_data], ws, 1, add_end=False)
        t = 0
        for x, xl, y, yl in bar:
            x = np.flip(x, axis=1)
            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print(ws.inverse_transform(x[0]))
            print(ws.inverse_transform(y[0]))
            print(ws.inverse_transform(pred[0]))
            t += 1
            if t >= 3:
                break


    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=1,
        **params
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        bar = batch_flow([x_data, y_data], ws, 1, add_end=False)
        t = 0
        for x, xl, y, yl in bar:
            x = np.flip(x, axis=1)
            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print(ws.inverse_transform(x[0]))
            print(ws.inverse_transform(y[0]))
            print(ws.inverse_transform(pred[0]))
            t += 1
            if t >= 3:
                break
def main():
    import json
    test(json.load(open('params.json')))

if __name__ == '__main__':
    main()