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
        if self.config.learning_decay:
            if self.config.start_decay_step:
                start_decay_step = self.config.start_decay_step
            else:
                start_decay_step = int(self.config.num_train_steps/2)
            remain_steps = self.config.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / self.config.decay_times)
            print("learning rate - {0}, start decay step - {1}, decay ratio - {2}, decay times - {3}".format(
                self.config.learning_rate, start_decay_step, self.config.decay_factor, self.config.decay_times))
            self.learning_rate = tf.cond(
                self.global_step < start_decay_step,
                lambda: self.learning_rate,
                lambda: tf.train.exponential_decay(
                    self.learning_rate,
                    (self.global_step - start_decay_step),
                    decay_steps, self.config.decay_factor, staircase=True),
                name="learning_rate_decay_cond")
        opt = get_optimizer(self.config.optimizer)(self.learning_rate)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.losses, params)
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
        tf.summary.scalar("learning_rate", self.learning_rate)
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

        if self.train_phase:
            self.keep_prob = tf.placeholder(tf.float32, name="Dropout")

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
                self.source_inputs = tf.nn.dropout(self.source_inputs, self.keep_prob)

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
            if self.train_phase:
                hidden = tf.nn.dropout(hidden, self.keep_prob)
            project_layer = tf.layers.Dense(self.config.num_tags,kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), name='project_layer')
            # batch size * sentence length * num_tags
            self.pred = project_layer(hidden)

    def setup_crf(self):
        with tf.variable_scope("CRF"):
            # 
            # According to [Neural architectures for named entity recognition](https://arxiv.org/pdf/1603.01360.pdf)
            # we add y_0 and y_n as start and end point
            # 
            small = -1000.0
            # start logits: batch_size * 1 * (num_tags + 2)
            # end logits: batch_size * 1 * (num_tags + 2)
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.config.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1]), small * tf.ones(shape=[self.batch_size,1,1])], axis=-1)
            end_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.config.num_tags]), small * tf.ones(shape=[self.batch_size,1,1]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1
            )

            num_steps = tf.shape(self.source_tokens)[-1]

            # pad logits to batch_size * num_steps * (num_tags + 2)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, num_steps, 2]), tf.float32)
            logits = tf.concat([self.pred, pad_logits], axis=-1)
            
            # pad logits to batch_size * (num_steps + 2) * (num_tags + 2)
            self.logits = tf.concat([start_logits, logits, end_logits], axis=1)

            # pad targets to batch_size * (num_steps + 2)
            # start + targets + end
            self.targets = tf.concat(
                [tf.cast(self.config.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.target_tokens, tf.cast((self.config.num_tags+1)*tf.ones([self.batch_size, 1]), tf.int32)], axis=-1)


            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.targets, self.source_length + 2)
            self.transition_params = trans_params
            self.losses = tf.reduce_mean(-log_likelihood)
            viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(self.logits, self.transition_params, self.source_length + 2)
            self.viterbi_sequence = viterbi_sequence[:,1:-1]


    def train_one_batch(self,source_tokens,source_length,segment_tokens,target_tokens):
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.segment_tokens] = segment_tokens
        feed_dict[self.target_tokens] = target_tokens
        feed_dict[self.keep_prob] = self.config.keep_prob

        lr, losses, summary, global_step, _ = self.sess.run([self.learning_rate, self.losses, self.summary_op, self.global_step, self.updates], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, global_step)
        return losses, global_step, lr

    def inference_np(self, source_tokens, source_length, segment_tokens):
        '''
        Decode in numpy way
        '''
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.segment_tokens] = segment_tokens
        if self.train_phase:
            feed_dict[self.keep_prob] = 1.0

        tf_unary_scores, tf_transition_params = self.sess.run([self.logits, self.transition_params],feed_dict=feed_dict)
        decode_sequnces, decode_score = [],[]
        for tf_unary_scores_, tf_sequence_length_  in zip(tf_unary_scores, source_length):
            # Remove padding.
            tf_unary_scores_ = tf_unary_scores_[:tf_sequence_length_]
            # Compute the highest score and its tag sequence.
            tf_viterbi_sequence, tf_viterbi_score = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, tf_transition_params)
            decode_sequnces.append(tf_viterbi_sequence[1:-1])
            decode_score.append(tf_viterbi_score)
        return decode_sequnces, decode_score

    def evaluate(self, source_tokens, source_length, segment_tokens, target_tokens):
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.segment_tokens] = segment_tokens
        feed_dict[self.target_tokens] = target_tokens
        feed_dict[self.keep_prob] = 1.0
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
        if self.train_phase:
            feed_dict[self.keep_prob] = 1.0
            
        tf_viterbi_sequence, tf_viterbi_score = self.sess.run([self.viterbi_sequence, self.viterbi_score], feed_dict=feed_dict)
        return tf_viterbi_sequence, tf_viterbi_score