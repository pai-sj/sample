from keras.applications.vgg16 import VGG16
import tensorflow as tf
import numpy as np
import keras.backend as K


class EAST:
    """
    Building TF Graph & Session for Text Detection Model, EAST

    Order
    1. _attach_stem_network()
    2. _attach_branch_network()
    3. _attach_output_network()
    4. _attach_loss_network()
    5. _attach_optimizer()

    """
    def __init__(self):
        K.clear_session()
        self._session = K.get_session()
        self.graph = self._session.graph
        self._to_build = ['stem',
                          'branch',
                          'output',
                          'loss',
                          'optimizer']
        self._built = []

        self._initialize_placeholders()

    def build_graph(self):
        """
        EAST의 Tensorflow Graph를 구성함

        :return:
        self
        """
        return (self._attach_stem_network()
                ._attach_branch_network()
                ._attach_output_network()
                ._attach_loss_network()
                ._attach_optimizer())

    def initialize_variables(self):
        with self.graph.as_default():
            global_vars = tf.global_variables()

            is_not_initialized = self._session.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

            if len(not_initialized_vars):
                self._session.run(tf.variables_initializer(not_initialized_vars))

    def _initialize_placeholders(self):
        with self.graph.as_default():
            self._x = None # After
            self._y_true_cls = tf.placeholder(tf.float32,
                                              shape=(None, None, None, 1),
                                              name='y_true_cls')
            self._y_true_geo = tf.placeholder(tf.float32,
                                              shape=(None, None, None, 5),
                                              name='y_true_geo')
            self._lr = tf.placeholder_with_default(0.001, (),
                                                   name='learning_rate')
            tf.add_to_collection('inputs', self._y_true_cls)
            tf.add_to_collection('inputs', self._y_true_geo)
            tf.add_to_collection('inputs', self._lr)



    def _attach_stem_network(self):
        if 'stem' in self._built:
            print("stem network is already built")
            return self

        with tf.variable_scope('stem'):
            vgg16 = VGG16(include_top=False)

        with self.graph.as_default():
            self._x = vgg16.input

            self.feature_maps = []
            for i in range(5, 1, -1):
                feature_map = vgg16.get_layer('block{}_pool'.format(i))
                feature_tensor = tf.identity(feature_map.output,
                                             "f{}".format(6 - i))
                self.feature_maps.append(feature_tensor)

        tf.add_to_collection('inputs', self._x)
        self._built.append(self._to_build.pop(0))
        return self

    def _attach_branch_network(self, num_layers=(128, 64, 32, 32)):
        if 'branch' in self._built:
            print("branch network is already built")
            return self
        elif not 'branch' == self._to_build[0]:
            raise IndexError("you should build {} network".format(self._to_build[0]))

        def unpool(tensor):
            with tf.variable_scope('unpool'):
                shape = tf.shape(tensor)
                return tf.image.resize_bilinear(tensor,
                                                size=[shape[1] * 2, shape[2] * 2])

        with self.graph.as_default():
            f = None
            h = None
            g = None
            with tf.variable_scope('branch'):
                for i, f in enumerate(self.feature_maps):
                    num_layer = num_layers[i]
                    with tf.variable_scope('block{}'.format(i+1)):
                        if i == 0:
                            h = f
                        else:
                            concat = tf.concat([g, f], axis=-1)
                            squeeze = tf.layers.Conv2D(num_layer, (1, 1),
                                                       padding='same',
                                                       activation=tf.nn.relu,
                                                       name='conv_1x1')(concat)
                            h = tf.layers.Conv2D(num_layer, (3, 3),
                                                 padding='same',
                                                 activation=tf.nn.relu,
                                                 name='conv_3x3')(squeeze)
                        if i <= 2:
                            g = unpool(h)
                        else:
                            g = tf.layers.Conv2D(num_layer, (3, 3),
                                                 padding='same',
                                                 activation=tf.nn.relu)(h)

            self._branch_map = tf.identity(g, name='branch_map')

        self._built.append(self._to_build.pop(0))
        return self

    def _attach_output_network(self, text_scale=512):
        if 'output' in self._built:
            print("output network is already built")
            return self
        elif not 'output' == self._to_build[0]:
            raise IndexError("you should build {} network".format(self._to_build[0]))

        with self.graph.as_default():
            with tf.variable_scope('output'):
                score_map = tf.layers.Conv2D(1, (1, 1),
                                             activation=tf.nn.sigmoid,
                                             name='score')(self._branch_map)
                loc_map = tf.layers.Conv2D(4, (1, 1),
                                           activation=tf.nn.sigmoid)(self._branch_map)
                loc_map = tf.identity(text_scale * loc_map, name='location')

                with tf.variable_scope('angle'):
                    # angle should be in [-45, 45]
                    angle_map = tf.layers.Conv2D(1, (1, 1),
                                                 activation=tf.nn.sigmoid)(self._branch_map)
                    angle_map = (angle_map - 0.5) * np.pi / 2

                self._y_pred_cls = score_map
                self._y_pred_geo = tf.concat([loc_map, angle_map], axis=-1,
                                             name='geometry')
            self.graph.add_to_collection('outputs', self._y_pred_cls)
            self.graph.add_to_collection('outputs', self._y_pred_geo)

        self._built.append(self._to_build.pop(0))
        return self

    def _attach_loss_network(self,
                             iou_smooth=1e-5,
                             alpha_theta=10,
                             alpha_geo=1):
        if 'loss' in self._built:
            print("loss network is already built")
            return self
        elif not 'loss' == self._to_build[0]:
            raise IndexError("you should build {} network".format(self._to_build[0]))

        epsilon = 1e-7
        with self.graph.as_default():
            with tf.variable_scope("losses"):
                with tf.variable_scope('score'):
                    with tf.variable_scope('balance_factor'):
                        num_pos = tf.count_nonzero(self._y_true_cls, axis=[1, 2, 3], dtype=tf.float32)
                        num_tot = tf.reduce_prod(tf.shape(self._y_true_cls)[1:])
                        beta = num_pos / tf.cast(num_tot, tf.float32)
                        beta = tf.reshape(beta, shape=(-1, 1, 1, 1))

                    with tf.variable_scope('balanced_cross_entropy'):
                        bcse = -(beta * self._y_true_cls * tf.log(epsilon + self._y_pred_cls) +
                                 (1. - beta) * (1. - self._y_true_cls) * tf.log(epsilon + 1. - self._y_pred_cls))

                score_loss = tf.reduce_mean(bcse, name='score_loss')

                with tf.variable_scope('geometry'):
                    geo_mask = tf.identity(self._y_true_cls, name='geo_mask')
                    num_pos = tf.count_nonzero(self._y_true_cls, axis=[1, 2, 3], dtype=tf.float32, name='num_pos')

                    with tf.variable_scope('split_tensor'):
                        top_true, right_true, bottom_true, left_true, theta_true = tf.split(self._y_true_geo, 5, axis=3)
                        top_pred, right_pred, bottom_pred, left_pred, theta_pred = tf.split(self._y_pred_geo, 5, axis=3)

                    with tf.variable_scope('aabb'):
                        with tf.variable_scope("area"):
                            area_true = (top_true + bottom_true) * (right_true + left_true)
                            area_pred = (top_pred + bottom_pred) * (right_pred + left_pred)

                            w_intersect = (tf.minimum(right_true, right_pred)
                                           + tf.minimum(left_true, left_pred))
                            h_intersect = (tf.minimum(top_true, top_pred)
                                           + tf.minimum(bottom_true, bottom_pred))
                            area_intersect = w_intersect * h_intersect
                            area_union = area_true + area_pred - area_intersect

                        with tf.variable_scope('iou_loss'):
                            area_loss = -tf.log((area_intersect + iou_smooth)
                                                / (area_union + iou_smooth))
                            # geo_mask에서 1인 부분들만 학습에 들어감
                            # 전체 평균이 아닌 geo_mask에서 1인 것들만 학습하므로, num_pos로 나누어주어야 함
                            # 배치 별 로스의 합 / 배치 당 데이터의 수
                            area_loss = tf.reduce_sum(area_loss * geo_mask, axis=[1, 2, 3]) / num_pos

                    area_loss = tf.reduce_mean(area_loss, name='area_loss')

                    with tf.variable_scope('theta'):
                        angle_loss = (1 - tf.cos(theta_pred - theta_true))
                        # geo_mask에서 1인 부분들만 학습에 들어감
                        # 전체 평균이 아닌 geo_mask에서 1인 것들만 학습하므로, num_pos로 나누어주어야 함
                        # 배치 별 로스의 합 / 배치 당 데이터의 수
                        angle_loss = tf.reduce_sum(angle_loss * geo_mask, axis=[1, 2, 3]) / num_pos
                    angle_loss = tf.reduce_mean(angle_loss, name='angle_loss')

                    with tf.variable_scope('aabb_theta'):
                        geo_loss = area_loss + alpha_theta * angle_loss
                geo_loss = tf.identity(geo_loss, name='geo_loss')

                with tf.variable_scope('total_loss'):
                    loss = score_loss + alpha_geo * geo_loss
            self._loss = tf.identity(loss, name='loss')

            tf.add_to_collection(tf.GraphKeys.LOSSES, self._loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, score_loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, geo_loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, area_loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, angle_loss)

            tf.summary.scalar('loss', self._loss)
            tf.summary.scalar('area_loss', area_loss)
            tf.summary.scalar('angle_loss', angle_loss)
            tf.summary.scalar('geo_loss', geo_loss)
            tf.summary.scalar('score_loss', score_loss)

        self._built.append(self._to_build.pop(0))
        return self

    def _attach_optimizer(self):
        if 'optimizer' in self._built:
            print("optimizer network is already built")
            return self
        elif not 'optimizer' == self._to_build[0]:
            raise IndexError("you should build {} network".format(self._to_build[0]))

        with self.graph.as_default():
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            without_stem_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                       scope='^((?!stem).)*$')
            with tf.control_dependencies(update_ops):
                self._headtune_op = (tf.train
                               .AdamOptimizer(self._lr)
                               .minimize(self._loss,
                                         var_list=without_stem_variables,
                                         name='headtune_op'))
                self._finetune_op = (tf.train
                                     .AdamOptimizer(self._lr)
                                     .minimize(self._loss,
                                               name='finetune_op'))

        self._built.append(self._to_build.pop(0))
        return self
