import tensorflow as tf
from utils.model_utils import add_emb_vis, load_pretrained_emb_from_txt, get_optimizer, read_vocab
from utils.rnn_utils import bidirection_rnn_cell
import os 

class NERModel(object):
    def __init__(self, sess, config):
        assert config.mode in ["train", "eval", "inference"]
        self.config = config
        self.train_phase = config.mode == 'train'
        self.sess = sess

    def build(self):
        print("build graph")
        self.global_step = tf.Variable(0, trainable=False)
        self.setup_input_placeholders()
        self.setup_embedding()
        self.setup_bilstm()
        self.setup_projection()
        self.setup_crf()
        if self.train_phase:
            self.setup_train()
            self.setup_summary()
        self.saver = tf.train.Saver(tf.global_variables())

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def save_model(self, checkpoint_dir=None, epoch=None):
        if epoch is None:
            self.saver.save(self.sess, os.path.join(self.config.checkpoint_dir if not checkpoint_dir else checkpoint_dir,
                            "model.ckpt"), global_step=self.global_step)
        else:
            self.saver.save(self.sess, os.path.join(self.config.checkpoint_dir if not checkpoint_dir else checkpoint_dir,
                            "model.ckpt"), global_step=epoch)

    def restore_model(self, checkpoint_dir=None, epoch=None):
        if epoch is None:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(
                self.config.checkpoint_dir if not checkpoint_dir else checkpoint_dir))
        else:
            self.saver.restore(
                self.sess, os.path.join(self.config.checkpoint_dir if not checkpoint_dir else checkpoint_dir, "model.ckpt" + ("-%d" % epoch)))

    def setup_train(self):
        print("set up training")
        self.learning_rate = tf.constant(self.config.learning_rate)
        opt = get_optimizer(self.config.optimizer)(self.learning_rate)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.losses, params, colocate_gradients_with_ops=True)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.config.max_gradient_norm)
        self.gradient_norm = tf.global_norm(gradients)
        self.param_norm = tf.global_norm(params)
        self.updates = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

    def setup_summary(self):
        self.summary_writer = tf.summary.FileWriter(
            self.config.checkpoint_dir, self.sess.graph)
        tf.summary.scalar("train_loss", self.losses)
        tf.summary.scalar('gN',self.gradient_norm)
        tf.summary.scalar('pN',self.param_norm)
        self.summary_op = tf.summary.merge_all()

    def setup_input_placeholders(self):
        # batch size * sentence length
        self.source_tokens = tf.placeholder(tf.int32, shape=[None,None], name="source_tokens")
        # batch size
        self.source_length = tf.placeholder(tf.int32, shape=[None], name="source_length")
        # batch size * sentence length
        self.segment_tokens = tf.placeholder(tf.int32, shape=[None,None], name="segment_tokens")
        # batch size * sentece length
        self.target_tokens = tf.placeholder(tf.int32, shape=[None,None], name="target_tokens")

        self.batch_size = tf.shape(self.source_tokens)[0]

    def setup_embedding(self):
        with tf.variable_scope("Embedding"), tf.device("/cpu:0"):
            self.word2id,self.id2word = read_vocab(self.config.vocab_file)
            if self.config.pretrained_embedding_file:
                embedding = load_pretrained_emb_from_txt(self.id2word,self.config.pretrained_embedding_file)
                self.source_embedding = tf.get_variable("source_embedding",dtype=tf.float32,initializer=tf.constant(embedding))
            else:
                self.source_embedding = tf.get_variable("source_embedding",[len(self.id2word),self.config.embedding_size],dtype=tf.float32,initializer=tf.random_uniform_initializer(-1, 1))

            # write encode embedding to tensorboard
            if self.train_phase:
                add_emb_vis(self.source_embedding, self.id2word, self.config.checkpoint_dir)

            # batch size * sentence length * embedding size
            self.source_inputs = tf.nn.embedding_lookup(self.source_embedding, self.source_tokens)

            if self.config.segement_embedding_size:
                with tf.variable_scope("Segment_Embedding"), tf.device("/cpu:0"):
                    if self.config.segment_onehot:
                        segment_inputs = tf.one_hot(self.segment_tokens, self.config.segment_tag_num)
                    else:
                        self.segment_embedding = tf.get_variable("segment_embedding",[self.config.segment_tag_num,self.config.segement_embedding_size],dtype=tf.float32,initializer=tf.random_uniform_initializer(-1, 1))
                        segment_inputs = tf.nn.embedding_lookup(self.segment_embedding, self.segment_tokens)

                self.source_inputs = tf.concat([self.source_inputs, segment_inputs], axis=-1)

            if self.train_phase:
                self.source_inputs = tf.nn.dropout(self.source_inputs, self.config.keep_prob)

    def setup_bilstm(self):
        with tf.variable_scope("BILSTM"):
            bi_encoder_outputs, bi_encoder_state = bidirection_rnn_cell(self.config.encode_cell_type, self.config.num_units, self.config.num_bi_layers, self.train_phase,
                                                                        self.config.keep_prob, self.config.num_gpus, self.source_length, self.source_inputs)

            # batch size * sentence length * (2 *num_units)
            self.encode_outputs = bi_encoder_outputs
            self.encode_state = bi_encoder_state

    def setup_projection(self):
        with tf.variable_scope("Projection"):
            hidden_layer = tf.layers.Dense(self.config.num_units,tf.tanh,kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), name='hidden_layer')
            hidden = hidden_layer(self.encode_outputs)
            project_layer = tf.layers.Dense(self.config.num_tags,kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), name='project_layer')
            # batch size * sentence length * num_tags
            self.pred = project_layer(hidden)

    def setup_crf(self):
        with tf.variable_scope("CRF"):
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.pred,self.target_tokens,self.source_length)
            self.transition_params = trans_params
            self.losses = tf.reduce_mean(-log_likelihood)
            self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(self.pred, self.transition_params, self.source_length)


    def train_one_batch(self,source_tokens,source_length,segment_tokens,target_tokens):
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.segment_tokens] = segment_tokens
        feed_dict[self.target_tokens] = target_tokens

        losses, summary, global_step, _ = self.sess.run([self.losses, self.summary_op, self.global_step, self.updates], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, global_step)
        return losses, global_step

    def inference_np(self, source_tokens, source_length, segment_tokens):
        '''
        Decode in numpy way
        '''
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.segment_tokens] = segment_tokens

        tf_unary_scores, tf_transition_params = sess.run([self.pred, self.transition_params],feed_dict=feed_dict)
        decode_sequnces, decode_score = [],[]
        for tf_unary_scores_, tf_sequence_length_  in zip(tf_unary_scores, source_length):
            # Remove padding.
            tf_unary_scores_ = tf_unary_scores_[:tf_sequence_length_]
            # Compute the highest score and its tag sequence.
            tf_viterbi_sequence, tf_viterbi_score = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, tf_transition_params)
            decode_sequnces.append(tf_viterbi_sequence)
            decode_score.append(tf_viterbi_score)
        return decode_sequnces, decode_score

    def evaluate(self, source_tokens, source_length, segment_tokens, target_tokens):
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.segment_tokens] = segment_tokens
        feed_dict[self.target_tokens] = target_tokens
        tf_viterbi_sequence, tf_viterbi_score = self.sess.run([self.viterbi_sequence, self.viterbi_score], feed_dict=feed_dict)
        return source_tokens, source_length, tf_viterbi_sequence, target_tokens

    def inference(self, source_tokens, source_length, segment_tokens):
        '''
        Decode inside graph
        '''
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.segment_tokens] = segment_tokens

        tf_viterbi_sequence, tf_viterbi_score = self.sess.run([self.viterbi_sequence, self.viterbi_score], feed_dict=feed_dict)
        return tf_viterbi_sequence, tf_viterbi_score