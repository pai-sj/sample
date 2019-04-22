from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
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
    4. _attach_decode_network()
    4. _attach_loss_network()
    5. _attach_optimizer()

    """

    def __init__(self):
        """

        east = EAST()

        Building Order
        1. east._attach_stem_network()
        2. east._attach_branch_network()
        3. east._attach_output_network()
        4. east._attach_decode_network()
        4. east._attach_loss_network()
        5. east._attach_optimizer()

        Intializing Variable
        east.initialize_variable()

        """
        K.clear_session()
        self.session = K.get_session() # Keras Pretrained Model을 쓰기 위함
        self.graph = self.session.graph

        # Layer을 쌓는 순서를 결정해줌
        self._to_build = ['stem',
                          'branch',
                          'output',
                          'decode',
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
                ._attach_decode_network()
                ._attach_loss_network()
                ._attach_optimizer())

    def initialize_variables(self):
        with self.graph.as_default():
            global_vars = tf.global_variables()

            is_not_initialized = self.session.run(
                [tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [
                v for (
                    v,
                    f) in zip(
                    global_vars,
                    is_not_initialized) if not f]

            if len(not_initialized_vars):
                self.session.run(
                    tf.variables_initializer(not_initialized_vars))

    def _initialize_placeholders(self):
        with self.graph.as_default():
            self._x = None  # setup After building stem network
            self._is_train = tf.placeholder_with_default(False, (), name='is_train')

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

    def _attach_stem_network(self, base_model='vgg'):
        if 'stem' in self._built:
            print("stem network is already built")
            return self
        with tf.variable_scope('stem'):
            if base_model == "vgg":
                vgg16 = VGG16(include_top=False)
                with self.graph.as_default():
                    self._x = vgg16.input

                    self.feature_maps = []
                    for i in range(5, 1, -1):
                        feature_map = vgg16.get_layer('block{}_pool'.format(i))
                        feature_tensor = tf.identity(feature_map.output,
                                                     "f{}".format(6 - i))
                        self.feature_maps.append(feature_tensor)
            elif base_model == "resnet":
                resnet = ResNet50(include_top=False)
                """
                RESNET에 있는 Batch Normalization의 Mean & Average은 ImageNet의 데이터셋에
                학습된 Mean & Average. 이를 Training 단계와 Test 단계로 나누어서 학습하게 되면,
                기존 데이터셋과 충돌나게 됨.
                
                즉, 성능이 굉장히 드랍되는 효과가 발생함.  
                reference : https://github.com/keras-team/keras/issues/7177
                """
                with self.graph.as_default():
                    self._x = resnet.input

                    self.feature_maps = []
                    for i, layer_idx in zip(range(5, 1, -1),
                                            [49, 40, 22, 10]):
                        feature_map = resnet.get_layer(
                            'activation_{}'.format(layer_idx))
                        feature_tensor = tf.identity(feature_map.output,
                                                     "f{}".format(6 - i))
                        # Version에 따라서
                        # keras._version__ <= 2.2 인경우,
                        # stem/f4:0에
                        # tf.keras.backend.spatial_2d_padding
                        # 으로 padding을 붙여주어야 함
                        self.feature_maps.append(feature_tensor)
            else:
                raise ValueError("stem network should be one of them, 'vgg' or 'resnet'")

        self.graph.add_to_collection('inputs', self._x)
        self.graph.add_to_collection('inputs', self._is_train)

        self._built.append(self._to_build.pop(0))
        return self

    def _attach_branch_network(self, num_layers=(128, 64, 32, 32)):
        if 'branch' in self._built:
            print("branch network is already built")
            return self
        elif not 'branch' == self._to_build[0]:
            raise IndexError(
                "you should build {} network".format(
                    self._to_build[0]))

        def unpool(tensor):
            with tf.variable_scope('unpool'):
                shape = tf.shape(tensor)
                return tf.image.resize_bilinear(
                    tensor, size=[shape[1] * 2, shape[2] * 2])

        with self.graph.as_default():
            conv2d = tf.layers.Conv2D
            batch_norm = tf.layers.BatchNormalization
            with tf.variable_scope('branch'):
                for i, f in enumerate(self.feature_maps):
                    num_layer = num_layers[i]
                    with tf.variable_scope('block{}'.format(i + 1)):
                        if i == 0:
                            h = f
                        else:
                            concat = tf.concat([g, f], axis=-1)
                            s = conv2d(num_layer, (1, 1),
                                       padding='same',
                                       activation=tf.nn.relu,
                                       name='conv_1x1')(concat)
                            s = batch_norm()(s, training=self._is_train)
                            h = conv2d(num_layer, (3, 3),
                                       padding='same',
                                       activation=tf.nn.relu,
                                       name='conv_3x3')(s)
                            h = batch_norm()(h, training=self._is_train)
                        if i <= 2:
                            g = unpool(h)
                        else:
                            g = conv2d(num_layer, (3, 3),
                                       padding='same',
                                       activation=tf.nn.relu)(h)
                            g = batch_norm()(g, training=self._is_train)
            self._branch_map = tf.identity(g, name='branch_map')

        self._built.append(self._to_build.pop(0))
        return self

    def _attach_output_network(self, text_scale=512):
        if 'output' in self._built:
            print("output network is already built")
            return self
        elif not 'output' == self._to_build[0]:
            raise IndexError(
                "you should build {} network".format(
                    self._to_build[0]))

        with self.graph.as_default():
            conv2d = tf.layers.Conv2D
            with tf.variable_scope('output'):
                score_map = conv2d(1, (1, 1),
                                   activation=tf.nn.sigmoid,
                                   name='score')(self._branch_map)

                loc_map = conv2d(4, (1, 1),
                                 activation=tf.nn.sigmoid)(self._branch_map)
                loc_map = tf.identity(text_scale * loc_map, name='location')

                with tf.variable_scope('angle'):
                    # angle should be in [-45, 45]
                    angle_map = conv2d(1, (1, 1),
                                       activation=tf.nn.sigmoid)(self._branch_map)
                    angle_map = (angle_map - 0.5) * np.pi / 2

            self._y_pred_cls = tf.identity(score_map, name='score')
            self._y_pred_geo = tf.concat([loc_map, angle_map], axis=-1, name='geometry')

            self.graph.add_to_collection('outputs', self._y_pred_cls)
            self.graph.add_to_collection('outputs', self._y_pred_geo)

        self._built.append(self._to_build.pop(0))
        return self

    def _attach_decode_network(self, fm_scale=4):
        if 'decode' in self._built:
            print("decode network is already built")
            return self
        elif not 'decode' == self._to_build[0]:
            raise IndexError(
                "you should build {} network".format(
                    self._to_build[0]))

        with self.graph.as_default():
            threshold = tf.placeholder_with_default(0.5, (), name='threshold')

            def decode_result(result_map):
                score_map, geo_map = tf.split(result_map, [1, 5], axis=2)
                score_map = score_map[:, :, 0]
                with tf.variable_scope('decoder'):
                    h, w, _ = tf.split(tf.shape(geo_map), 3)
                    h = tf.squeeze(h)
                    w = tf.squeeze(w)

                    xs, ys = tf.meshgrid(tf.range(0, w * fm_scale, fm_scale),
                                         tf.range(0, h * fm_scale, fm_scale), )
                    coords = tf.stack([ys, xs], axis=-1)
                    coords = tf.cast(coords, tf.float32)

                    indices = tf.where(score_map >= threshold)
                    exist_score_map = tf.gather_nd(score_map, indices)
                    exist_coords = tf.gather_nd(coords, indices)
                    exist_geo_map = tf.gather_nd(geo_map, indices)

                    p_y, p_x = tf.split(exist_coords, 2, axis=1)
                    top, right, bottom, left, theta = tf.split(exist_geo_map, 5, axis=1)

                    top_y = p_y - top
                    bot_y = p_y + bottom
                    left_x = p_x - left
                    right_x = p_x + right

                    tl = tf.concat([left_x, top_y], axis=1)
                    tr = tf.concat([right_x, top_y], axis=1)
                    br = tf.concat([right_x, bot_y], axis=1)
                    bl = tf.concat([left_x, bot_y], axis=1)

                    center_x = tf.reduce_mean([left_x, right_x], axis=0)
                    center_y = tf.reduce_mean([top_y, bot_y], axis=0)
                    center = tf.concat([center_x, center_y], axis=1)

                    shift_tl = tl - center
                    shift_tr = tr - center
                    shift_bl = bl - center
                    shift_br = br - center

                    theta = theta[:, 0]
                    x_rot_matrix = tf.stack([tf.cos(theta), -tf.sin(theta)], axis=1)
                    y_rot_matrix = tf.stack([tf.sin(theta), tf.cos(theta)], axis=1)

                    rotated_tl_x = (tf.reduce_sum(x_rot_matrix * shift_tl, axis=1)
                                    + center[:, 0])
                    rotated_tl_y = (tf.reduce_sum(y_rot_matrix * shift_tl, axis=1)
                                    + center[:, 1])

                    rotated_tr_x = (tf.reduce_sum(x_rot_matrix * shift_tr, axis=1)
                                    + center[:, 0])
                    rotated_tr_y = (tf.reduce_sum(y_rot_matrix * shift_tr, axis=1)
                                    + center[:, 1])

                    rotated_br_x = (tf.reduce_sum(x_rot_matrix * shift_br, axis=1)
                                    + center[:, 0])
                    rotated_br_y = (tf.reduce_sum(y_rot_matrix * shift_br, axis=1)
                                    + center[:, 1])

                    rotated_bl_x = (tf.reduce_sum(x_rot_matrix * shift_bl, axis=1)
                                    + center[:, 0])
                    rotated_bl_y = (tf.reduce_sum(y_rot_matrix * shift_bl, axis=1)
                                    + center[:, 1])

                    rotated_tl = tf.stack([rotated_tl_x, rotated_tl_y], axis=-1)
                    rotated_tr = tf.stack([rotated_tr_x, rotated_tr_y], axis=-1)
                    rotated_bl = tf.stack([rotated_bl_x, rotated_bl_y], axis=-1)
                    rotated_br = tf.stack([rotated_br_x, rotated_br_y], axis=-1)

                    rotated_polys = tf.stack([rotated_tl, rotated_tr,
                                              rotated_br, rotated_bl], axis=1)

                rotated_polys = tf.identity(rotated_polys, name="selected_polys")
                exist_score_map = tf.identity(exist_score_map, name='selected_scores')

                rotated_polys = tf.reshape(rotated_polys, (-1, 8))
                exist_score_map = tf.expand_dims(exist_score_map, axis=1)
                result_map = tf.concat([rotated_polys, exist_score_map], axis=-1)

                return result_map

            results = tf.concat([self._y_pred_cls, self._y_pred_geo], axis=-1)
            decode = tf.map_fn(decode_result, results, name='decode')

            self.graph.add_to_collection('decode', decode)
        self._built.append(self._to_build.pop(0))
        return self

    def _attach_loss_network(self,
                             loss_type='bcse',
                             iou_smooth=1e-5,
                             alpha_theta=10,
                             alpha_geo=1):
        if 'loss' in self._built:
            print("loss network is already built")
            return self
        elif not 'loss' == self._to_build[0]:
            raise IndexError(
                "you should build {} network".format(
                    self._to_build[0]))

        epsilon = 1e-7
        with self.graph.as_default():
            with tf.variable_scope("losses"):
                with tf.variable_scope('score'):
                    if loss_type == "bcse":
                        with tf.variable_scope('balance_factor'):
                            num_pos = tf.count_nonzero(self._y_true_cls,
                                                       axis=[1, 2, 3],
                                                       dtype=tf.float32)
                            num_tot = tf.reduce_prod(
                                tf.shape(self._y_true_cls)[1:])
                            beta = 1 - num_pos / tf.cast(num_tot, tf.float32)
                            beta = tf.reshape(beta, shape=(-1, 1, 1, 1))

                        with tf.variable_scope('balanced_cross_entropy'):
                            bcse = -(beta * self._y_true_cls * tf.log(epsilon + self._y_pred_cls) +
                                     (1. - beta) * (1. - self._y_true_cls) * tf.log(epsilon + 1. - self._y_pred_cls))
                            bcse = tf.reduce_sum(bcse, axis=[1, 2, 3])
                        score_loss = tf.reduce_mean(bcse, name='score_loss')

                    elif loss_type == "dice":
                        with tf.variable_scope('dice_coefficient'):
                            intersection = tf.reduce_sum(
                                self._y_true_cls * self._y_pred_cls)
                            union = tf.reduce_sum(
                                self._y_true_cls) + tf.reduce_sum(self._y_pred_cls) + epsilon
                            dice = 1 - 2 * intersection / union
                        score_loss = tf.identity(dice, name='score_loss')
                    else:
                        raise ValueError(
                            "loss_type should be one, 'dice', 'bcse'")

                with tf.variable_scope('geometry'):
                    geo_mask = tf.identity(self._y_true_cls, name='geo_mask')

                    with tf.variable_scope('split_tensor'):
                        top_true, right_true, bottom_true, left_true, theta_true = tf.split(
                            self._y_true_geo, 5, axis=3)
                        top_pred, right_pred, bottom_pred, left_pred, theta_pred = tf.split(
                            self._y_pred_geo, 5, axis=3)

                    with tf.variable_scope('aabb'):
                        with tf.variable_scope("area"):
                            area_true = (top_true + bottom_true) * \
                                (right_true + left_true)
                            area_pred = (top_pred + bottom_pred) * \
                                (right_pred + left_pred)

                            w_intersect = (tf.minimum(right_true, right_pred)
                                           + tf.minimum(left_true, left_pred))
                            h_intersect = (tf.minimum(top_true, top_pred) +
                                           tf.minimum(bottom_true, bottom_pred))
                            area_intersect = w_intersect * h_intersect
                            area_union = area_true + area_pred - area_intersect

                        with tf.variable_scope('iou_loss'):
                            area_loss = -tf.log((area_intersect + iou_smooth)
                                                / (area_union + iou_smooth))
                            # geo_mask에서 1인 부분들만 학습에 들어감
                            area_loss = tf.reduce_sum(
                                area_loss * geo_mask, axis=[1, 2, 3])

                    area_loss = tf.reduce_mean(area_loss, name='area_loss')

                    with tf.variable_scope('theta'):
                        angle_loss = (1 - tf.cos(theta_pred - theta_true))
                        # geo_mask에서 1인 부분들만 학습에 들어감
                        angle_loss = tf.reduce_sum(
                            angle_loss * geo_mask, axis=[1, 2, 3])
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

    def _attach_optimizer(self, weight_decay=1e-5):
        if 'optimizer' in self._built:
            print("optimizer network is already built")
            return self
        elif not 'optimizer' == self._to_build[0]:
            raise IndexError(
                "you should build {} network".format(
                    self._to_build[0]))

        with self.graph.as_default():
            with tf.variable_scope("optimizer"):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                without_stem_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='^((?!stem).)*$')

                with tf.variable_scope("l2_loss"):
                    weights = [var
                               for var in tf.trainable_variables()
                               if not "bias" in var.name]
                    l2_losses = tf.add_n([tf.nn.l2_loss(var) for var in weights], name='l2_losses')

                with tf.control_dependencies(update_ops):
                    loss = self._loss + weight_decay * l2_losses

                    self._headtune_op = (tf.train
                                         .AdamOptimizer(self._lr)
                                         .minimize(loss,
                                                   var_list=without_stem_variables,
                                                   name='headtune_op'))
                    self._finetune_op = (tf.train
                                         .AdamOptimizer(self._lr)
                                         .minimize(loss,
                                                   name='finetune_op'))

        self._built.append(self._to_build.pop(0))
        return self

