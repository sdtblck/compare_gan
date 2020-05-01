# coding=utf-8
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Self-Supervised GAN with contrastive loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
from compare_gan.architectures.arch_ops import linear
from compare_gan.gans import loss_lib
from compare_gan.gans import modular_gan
from compare_gan.gans import penalty_lib
from compare_gan.gans import utils

import gin
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS


random_brightness = gin.configurable(tf.image.random_brightness)
random_contrast = gin.configurable(tf.image.random_contrast)
random_saturation = gin.configurable(tf.image.random_saturation)
random_hue = gin.configurable(tf.image.random_hue)
clip_by_value = gin.configurable(tf.clip_by_value)
random_flip_left_right = gin.configurable(tf.image.random_flip_left_right)
random_flip_up_down = gin.configurable(tf.image.random_flip_up_down)

@gin.configurable
def noop(image):
  return image

def random_uniform(x, lower, upper, seed=None):
  if hasattr(x, 'shape'):
    shape = x.shape[0:1]
  else:
    shape = ()
  return tf.random.uniform(shape, lower, upper, seed=seed)

def lerp(a, b, t):
  return (b - a) * t + a

def random_lerp(a, b, lower, upper):
  t = random_uniform(a, lower, upper)
  return lerp(a, b, t)

@gin.configurable
def transform_image(image, target_image_shape, crop_method="distorted", seed=None,
    aspect_ratio_range=[4.0/3.0, 3.0/4.0],
    area_range=[0.08, 1.00],
    ):
  """Preprocesses ImageNet images to have a target image shape.

  Args:
    image: 3-D tensor with a single image.
    target_image_shape: List/Tuple with target image shape.
    crop_method: Method for cropping the image:
      One of: distorted, random, middle, none
    seed: Random seed, only used for `crop_method=distorted`.

  Returns:
    Image tensor with shape `target_image_shape`.
  """
  if crop_method == "distorted":
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        tf.zeros([0, 0, 4], tf.float32),
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        use_image_if_no_bounding_boxes=True,
        seed=seed)
    image = tf.slice(image, begin, size)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    image.set_shape([None, None, target_image_shape[-1]])
  elif crop_method == "random":
    tf.set_random_seed(seed)
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    size = tf.minimum(h, w)
    begin = [h - size, w - size] * tf.random.uniform([2], 0, 1)
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    image = tf.slice(image, begin, [size, size, 3])
  elif crop_method == "middle":
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    size = tf.minimum(h, w)
    begin = tf.cast([h - size, w - size], tf.float32) / 2.0
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    image = tf.slice(image, begin, [size, size, 3])
  elif crop_method != "none":
    raise ValueError("Unsupported crop method: {}".format(crop_method))
  image = tf.image.resize_images(
      image, [target_image_shape[0], target_image_shape[1]],
      method=tf.image.ResizeMethod.AREA)
  image.set_shape(target_image_shape)
  return image

@gin.configurable
def random_crop_and_resize(x, crop_method="distorted"):
  b, h, w, c = x.get_shape().as_list()
  crop = tf.map_fn(lambda x: transform_image(x, [h, w, c], crop_method), x)
  return crop

@gin.configurable
def distort_exposure(image, lower=1.0/2.2, upper=2.2, invert=False):
  if invert:
    lower = 1.0 / lower
    upper = 1.0 / upper
  t = random_uniform((), -1.0, 1.0)
  u = 1.0
  u = lerp(u, lower, tf.maximum(0.0, -t))
  u = lerp(u, upper, tf.maximum(0.0, t))
  return tf.image.adjust_gamma(image, gamma=u)

@gin.configurable
def distort_contrast(x, lower=0.5, upper=1.5):
  a = tf.image.adjust_contrast(x, contrast_factor=lower)
  b = tf.image.adjust_contrast(x, contrast_factor=upper)
  return random_lerp(a, b, 0.0, 1.0)

@gin.configurable
def distort_brightness(x, lower=0.8, upper=1.2):
  a = tf.image.adjust_brightness(x, delta=lower - 1.0)
  b = tf.image.adjust_brightness(x, delta=upper - 1.0)
  return random_lerp(a, b, 0.0, 1.0)

@gin.configurable
def distort_saturation(x, lower=0.5, upper=1.5):
  a = tf.image.adjust_saturation(x, saturation_factor=lower)
  b = tf.image.adjust_saturation(x, saturation_factor=upper)
  return random_lerp(a, b, 0.0, 1.0)

@gin.configurable
def distort_hue(x, max_delta=0.5):
  x = tf.image.random_hue(x, max_delta=max_delta)
  return x

@gin.configurable
def distort_sobel(image, lower=0.6, upper=1.0):
  grad_components = tf.image.sobel_edges(image)
  grad_mag_components = grad_components**2
  grad_mag_square = tf.math.reduce_sum(grad_mag_components,axis=-1)
  grad_mag_img = tf.sqrt(grad_mag_square)
  a = grad_mag_img
  b = image
  return random_lerp(a, b, lower, upper)

@gin.configurable
def sobel(image):
  image = tf.image.rgb_to_grayscale(image)
  grad_components = tf.image.sobel_edges(image)
  y = grad_components
  y = y**2
  y = tf.math.reduce_sum(y,axis=-1) # sum all magnitude components
  y = tf.sqrt(y)
  y = tf.image.grayscale_to_rgb(y)
  return y

@gin.configurable
def sobel_rgb(image):
  y = tf.image.sobel_edges(image)
  y = y**2
  y = tf.math.reduce_sum(y,axis=-1)
  y = tf.sqrt(y)
  return y

@gin.configurable
def color_drop(image):
  image = tf.image.rgb_to_grayscale(image)
  image = tf.image.grayscale_to_rgb(image)
  return image

# https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
def gaussian_kernel(size, mean=1.0, std=1.0):
  """Makes 2D gaussian Kernel for convolution."""
  d = tf.distributions.Normal(mean, std)
  vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
  gauss_kernel = tf.einsum('i,j->ij', vals, vals)
  return gauss_kernel / tf.reduce_sum(gauss_kernel)

@gin.configurable
def gaussian_blur(x, size=10.0, mean=10.0, std=4.0):
  # Make Gaussian Kernel with desired specs.
  gauss_kernel = gaussian_kernel(size=size, mean=mean, std=std)
  # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
  gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
  # Convolve.
  kernel = gauss_kernel
  yiq = tf.image.rgb_to_yiq(x)
  y = tf.slice(yiq, [0, 0, 0, 0], tf.concat([tf.shape(x)[0:3],[1]], axis=0))
  iq = tf.slice(yiq, [0, 0, 0, 1], tf.concat([tf.shape(x)[0:3],[2]], axis=0))
  y = tf.nn.conv2d(y, kernel, [1, 1, 1, 1], "SAME")
  yiq = tf.concat([y, iq], axis=3)
  x = tf.image.yiq_to_rgb(yiq)
  # Ensure it's not trainable.
  x = tf.stop_gradient(x)
  return x

@gin.configurable
def blur(x, image_size, percentage=0.10, size_max=None, std_min=0.1, std_max=2.0):
  k = percentage * image_size
  if size_max is not None:
    k = tf.minimum(k, size_max)
  return gaussian_blur(x, size=k, mean=1.0, std=random_uniform(x, std_min, std_max))

@gin.configurable
def motionblur(image, size_min=2, size_max=10):
  size = random_uniform((), size_min, size_max)
  size_2 = tf.cast(size / 2, tf.int32)
  size_1 = tf.cast(size, tf.int32)
  x = image
  kernel = tf.concat([tf.zeros((size_2, size_1)), [tf.ones((size_1))], tf.zeros((size_2, size_1))], axis=0) / size
  kernel = kernel[:, :, tf.newaxis, tf.newaxis]
  yiq = tf.image.rgb_to_yiq(x)
  y = tf.slice(yiq, [0, 0, 0, 0], tf.concat([tf.shape(x)[0:3],[1]], axis=0))
  iq = tf.slice(yiq, [0, 0, 0, 1], tf.concat([tf.shape(x)[0:3],[2]], axis=0))
  y = tf.nn.conv2d(y, kernel, [1, 1, 1, 1], "SAME")
  yiq = tf.concat([y, iq], axis=3)
  x = tf.image.yiq_to_rgb(yiq)
  # Ensure it's not trainable.
  x = tf.stop_gradient(x)
  return x

@gin.configurable
def resize(x, size, method=tf.image.ResizeMethod.AREA):
  if isinstance(size, int) or isinstance(size, float):
    size = [size, size]
  return tf.image.resize(x, size, method=method)

@gin.configurable
def resize_512(x):
  return resize(x, 512)

@gin.configurable
def resize_256(x):
  return resize(x, 256)

@gin.configurable
def resize_128(x):
  return resize(x, 128)

@gin.configurable
def resize_64(x):
  return resize(x, 64)

@gin.configurable
def resize_32(x):
  return resize(x, 32)

def call(f, x):
  if callable(f):
    x = f(x)
  else:
    for f1 in f:
      x = f1(x)
  return x

def call_dynamic(x, transforms, seed=None):
  if not isinstance(transforms, list):
    transforms = [transforms]
  for l in transforms:
    prob, f = [1.0, l] if not isinstance(l, list) else l
    if prob >= 1.0:
      if isinstance(f, list):
        x = call_dynamic(x, f)
      else:
        x = call(f, x)
    elif prob <= 0.0:
      pass
    else:
      x = call_maybe(prob, f, x, seed=seed)
  return x

def call_maybe(prob, f, x, seed=None):
  return tf.cond(
      tf.less_equal(random_uniform((), 0.0, 1.0, seed=seed), prob),
      lambda: call_dynamic(x, f),
      lambda: x)

# def call_maybe(prob, f, x, seed=None):
#   a = call_dynamic(x, f)
#   b = tf.identity(x)
#   selector = tf.cast(tf.less_equal(random_uniform(x, 0.0, 1.0, seed=seed), prob), tf.float32)
#   return a * selector + b * (1.0 - selector)

@gin.configurable
def augment(x, transforms=gin.REQUIRED, evaluate=call_dynamic):
  return evaluate(x, transforms)

# pylint: disable=not-callable
@gin.configurable(blacklist=["kwargs"])
class CLGAN(modular_gan.ModularGAN):
  """Self-Supervised GAN with Contrastive Loss"""

  def __init__(self,
               weight_contrastive_loss_d=10.0,
               **kwargs):
    """Creates a new Self-Supervised GAN using Contrastive Loss.

    Args:
      self_supervised_batch_size: The total number images per batch for the self supervised loss.
      weight_contrastive_loss_d: Weight for the contrastive loss for the self supervised learning on real images
      **kwargs: Additional arguments passed to `ModularGAN` constructor.
    """
    super(CLGAN, self).__init__(**kwargs)

    self._weight_contrastive_loss_d = weight_contrastive_loss_d

    # To safe memory ModularGAN supports feeding real and fake samples
    # separately through the discriminator. CLGAN does not support this to
    # avoid additional additional complexity in create_loss().
    assert not self._deprecated_split_disc_calls, \
        "Splitting discriminator calls is not supported in CLGAN."

  def _latent_projections(self, latents):
    bs, dim = latents.get_shape().as_list()

    with tf.variable_scope("discriminator_z_projection", reuse=tf.AUTO_REUSE) as scope:
      k1 = tf.get_variable("kernel1", [dim, dim * 4])
      k2 = tf.get_variable("kernel2", [dim * 4, dim])
      z_proj = tf.matmul(tf.nn.leaky_relu(tf.matmul(latents, k1), name=scope.name), k2)
      z_proj = z_proj / tf.reshape(tf.norm(z_proj, ord=2, axis=-1), [bs, 1])
      return z_proj

  def create_loss(self, features, labels, params, is_training=True):
    """Build the loss tensors for discriminator and generator.

    This method will set self.d_loss and self.g_loss.

    Args:
      features: Optional dictionary with inputs to the model ("images" should
          contain the real images and "z" the noise for the generator).
      labels: Tensor will labels. These are class indices. Use
          self._get_one_hot_labels(labels) to get a one hot encoded tensor.
      params: Dictionary with hyperparameters passed to TPUEstimator.
          Additional TPUEstimator will set 3 keys: `batch_size`, `use_tpu`,
          `tpu_context`. `batch_size` is the batch size for this core.
      is_training: If True build the model in training mode. If False build the
          model for inference mode (e.g. use trained averages for batch norm).

    Raises:
      ValueError: If set of meta/hyper parameters is not supported.
    """
    images = features["images"]  # Input images.
    generated = features["generated"]  # Fake images.
    if self.conditional:
      y = self._get_one_hot_labels(labels)
      sampled_y = self._get_one_hot_labels(features["sampled_labels"])
    else:
      y = None
      sampled_y = None
      all_y = None

    # Batch size per core.
    bs = images.shape[0].value

    with gin.config_scope("reals"):
      aug_images = augment(images)
      self._add_images_to_summary(aug_images, "real_images_aug", params)

    with gin.config_scope("fakes"):
      aug_generated = augment(generated)
      self._add_images_to_summary(aug_images, "fake_images_aug", params)

    # concat all images
    all_images = tf.concat([images, generated, aug_images, aug_generated], 0)

    if self.conditional:
      all_y = tf.concat([y, sampled_y, y, sampled_y], axis=0)

    # Compute discriminator output for real and fake images in one batch.

    d_all, d_all_logits, d_latents = self.discriminator(
        x=all_images, y=all_y, is_training=is_training)

    z_projs = self._latent_projections(d_latents)

    d_real, d_fake, _, _ = tf.split(d_all, 4)
    d_real_logits, d_fake_logits, _, _ = tf.split(d_all_logits, 4)
    z_projs_real, z_projs_fake, z_aug_projs_real, z_aug_projs_fake = tf.split(z_projs, 4)

    self.d_loss, _, _, self.g_loss = loss_lib.get_losses(
        d_real=d_real, d_fake=d_fake, d_real_logits=d_real_logits,
        d_fake_logits=d_fake_logits)

    penalty_loss = penalty_lib.get_penalty_loss(
        x=images, x_fake=generated, y=y, is_training=is_training,
        discriminator=self.discriminator, architecture=self._architecture)
    self.d_loss += self._lambda * penalty_loss

    z_projs = tf.concat([z_projs_real, z_projs_fake], 0)
    z_aug_projs = tf.concat([z_aug_projs_real, z_aug_projs_fake], 0)

    sims_logits = tf.matmul(z_projs, z_aug_projs, transpose_b=True)
    logits_max = tf.reduce_max(sims_logits, 1)
    sims_logits = sims_logits - tf.reshape(logits_max, [-1, 1])
    sims_probs = tf.nn.softmax(sims_logits)

    sim_labels = tf.constant(np.arange(bs * 2, dtype=np.int32))
    sims_onehot = tf.one_hot(sim_labels, bs * 2)

    c_real_loss = - tf.reduce_mean(
      tf.reduce_sum(sims_onehot * tf.log(sims_probs + 1e-10), 1))

    self.d_loss += c_real_loss * self._weight_contrastive_loss_d

    self._tpu_summary.scalar("loss/c_real_loss", c_real_loss)
    self._tpu_summary.scalar("loss/penalty", penalty_loss)

