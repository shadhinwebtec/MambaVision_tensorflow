import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from einops import rearrange,repeat
import math
import numpy as np

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    windows = tf.reshape(tf.transpose(x, (0, 1, 3, 2, 4, 5)), (-1, window_size * window_size, C))
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size*window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = tf.reshape(windows, (B, H // window_size, W // window_size, window_size, window_size, -1))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, (B, H, W, -1))
    return x

class PatchEmbed(layers.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, name="proj")

    def call(self, x):
        x = self.proj(x)
        x = tf.reshape(x, (-1, self.num_patches, x.shape[-1]))
        return x

class Mlp(layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.nn.gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features)
        self.act = act_layer
        self.fc2 = layers.Dense(out_features)
        self.drop = layers.Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class VisionTransformer(Model):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=layers.LayerNormalization):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = self.add_weight("cls_token", shape=(1, 1, embed_dim), initializer="zeros")
        self.pos_embed = self.add_weight("pos_embed", shape=(1, self.patch_embed.num_patches + 1, embed_dim), initializer="zeros")
        self.pos_drop = layers.Dropout(drop_rate)
        self.blocks = [layers.Dense(embed_dim, name=f"blocks_{i}") for i in range(depth)]
        self.norm = norm_layer()
        self.head = layers.Dense(num_classes, name="head") if num_classes > 0 else tf.identity

    def call(self, x):
        B = tf.shape(x)[0]
        x = self.patch_embed(x)
        cls_tokens = tf.broadcast_to(self.cls_token, [B, 1, self.embed_dim])
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

def _load_checkpoint(model, filename):
    """Load checkpoint from a file or URI."""
    checkpoint = tf.keras.models.load_model(filename)
    model.set_weights(checkpoint.get_weights())




class Downsample(layers.Layer):
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = layers.Conv2D(dim_out, kernel_size=3, strides=2, padding='same', use_bias=False)

    def call(self, x):
        x = self.reduction(x)
        return x

class PatchEmbed(layers.Layer):
    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = layers.Lambda(lambda x: x)
        self.conv_down = tf.keras.Sequential([
            layers.Conv2D(in_dim, kernel_size=3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(epsilon=1e-4),
            layers.ReLU(),
            layers.Conv2D(dim, kernel_size=3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(epsilon=1e-4),
            layers.ReLU()
        ])

    def call(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x

class ConvBlock(layers.Layer):
    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=3):
        super().__init__()
        self.conv1 = layers.Conv2D(dim, kernel_size=kernel_size, padding='same')
        self.norm1 = layers.BatchNormalization(epsilon=1e-5)
        self.act1 = layers.GELU(approximate='tanh')
        self.conv2 = layers.Conv2D(dim, kernel_size=kernel_size, padding='same')
        self.norm2 = layers.BatchNormalization(epsilon=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.gamma = self.add_weight(shape=(dim,), initializer=tf.constant_initializer(layer_scale), trainable=True)
        else:
            self.layer_scale = False
        self.drop_path = layers.Dropout(drop_path) if drop_path > 0. else layers.Lambda(lambda x: x)

    def call(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * tf.reshape(self.gamma, (1, 1, 1, -1))
        x = input + self.drop_path(x)
        return x

class MambaVisionMixer(layers.Layer):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = layers.Dense(self.d_inner, use_bias=bias)
        self.x_proj = layers.Dense(self.dt_rank + self.d_state * 2, use_bias=False)
        self.dt_proj = layers.Dense(self.d_inner // 2, use_bias=True)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            self.dt_proj.set_weights([tf.constant_initializer(dt_init_std)(shape=self.dt_proj.weights[0].shape), self.dt_proj.weights[1]])
        elif dt_init == "random":
            self.dt_proj.set_weights([tf.random.uniform(self.dt_proj.weights[0].shape, -dt_init_std, dt_init_std), self.dt_proj.weights[1]])
        else:
            raise NotImplementedError
        
        dt = tf.exp(
            tf.random.uniform(shape=(self.d_inner // 2,), minval=math.log(dt_min), maxval=math.log(dt_max))
        ).numpy()
        dt = tf.clip_by_value(dt, clip_value_min=dt_init_floor, clip_value_max=dt_max)
        inv_dt = dt + tf.math.log(-tf.math.expm1(-dt))
        self.dt_proj.bias.assign(inv_dt)

        A = repeat(
            tf.range(1, self.d_state + 1, dtype=tf.float32),
            "n -> d n",
            d=self.d_inner // 2,
        ).numpy()
        A_log = tf.math.log(A)
        self.A_log = tf.Variable(A_log, trainable=True)
        self.D = tf.Variable(tf.ones(self.d_inner // 2), trainable=True)
        self.out_proj = layers.Dense(self.d_model, use_bias=bias)
        self.conv1d_x = layers.Conv1D(
            filters=self.d_inner // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            use_bias=conv_bias // 2,
            padding='same'
        )
        self.conv1d_z = layers.Conv1D(
            filters=self.d_inner // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            use_bias=conv_bias // 2,
            padding='same'
        )

    def call(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        B, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = tf.split(xz, 2, axis=1)
        A = -tf.exp(self.A_log)
        x = tf.nn.silu(self.conv1d_x(x))
        z = tf.nn.silu(self.conv1d_z(z))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = tf.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], axis=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen)
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen)
        
        # `selective_scan_fn` needs to be implemented or replaced with an appropriate function
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D, 
                              z=None, 
                              delta_bias=self.dt_proj.bias, 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = tf.concat([y, z], axis=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out



    



class Downsample(layers.Layer):
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = layers.Conv2D(dim_out, 3, strides=2, padding='same', use_bias=False)

    def call(self, x):
        x = self.reduction(x)
        return x


class PatchEmbed(layers.Layer):
    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = layers.Identity()
        self.conv_down = tf.keras.Sequential([
            layers.Conv2D(in_dim, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(epsilon=1e-4),
            layers.ReLU(),
            layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(epsilon=1e-4),
            layers.ReLU()
        ])

    def call(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(layers.Layer):
    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=3):
        super().__init__()
        self.conv1 = layers.Conv2D(dim, kernel_size=kernel_size, padding='same')
        self.norm1 = layers.BatchNormalization(epsilon=1e-5)
        self.act1 = layers.Activation('gelu')
        self.conv2 = layers.Conv2D(dim, kernel_size=kernel_size, padding='same')
        self.norm2 = layers.BatchNormalization(epsilon=1e-5)
        self.layer_scale = layer_scale

        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.gamma = self.add_weight(name='gamma', shape=(dim,), initializer=tf.keras.initializers.Constant(layer_scale), trainable=True)
            self.layer_scale = True
        else:
            self.layer_scale = False

        self.drop_path = layers.Dropout(drop_path) if drop_path > 0. else layers.Lambda(lambda x: x)

    def call(self, x):
        input_tensor = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * tf.reshape(self.gamma, (1, 1, 1, -1))
        x = input_tensor + self.drop_path(x)
        return x


class Attention(layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=layers.LayerNormalization):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.q_norm = norm_layer() if qk_norm else layers.Lambda(lambda x: x)
        self.k_norm = norm_layer() if qk_norm else layers.Lambda(lambda x: x)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = tf.reshape(self.qkv(x), (B, N, 3, self.num_heads, self.head_dim))
        q, k, v = tf.unstack(qkv, axis=2)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = tf.matmul(attn, v)

        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(layers.Layer):
    def __init__(self, dim, num_heads, counter, transformer_blocks, mlp_ratio=4., qkv_bias=False, qk_scale=False, drop=0., attn_drop=0., drop_path=0., act_layer=layers.Activation, norm_layer=layers.LayerNormalization, Mlp_block=layers.Dense, layer_scale=None):
        super().__init__()
        self.norm1 = norm_layer()
        if counter in transformer_blocks:
            self.mixer = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_scale, attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer)
        else:
            self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)

        self.drop_path = layers.Dropout(drop_path) if drop_path > 0. else layers.Lambda(lambda x: x)
        self.norm2 = norm_layer()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = tf.keras.Sequential([
            Mlp_block(mlp_hidden_dim, activation=act_layer),
            layers.Dropout(drop)
        ])
        self.gamma_1 = self.add_weight(name='gamma_1', shape=(dim,), initializer=tf.keras.initializers.Constant(layer_scale), trainable=True) if layer_scale is not None and isinstance(layer_scale, (int, float)) else 1
        self.gamma_2 = self.add_weight(name='gamma_2', shape=(dim,), initializer=tf.keras.initializers.Constant(layer_scale), trainable=True) if layer_scale is not None and isinstance(layer_scale, (int, float)) else 1

    def call(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(layers.Layer):
    def __init__(self, dim, depth, num_heads, window_size, conv=False, downsample=True, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., layer_scale=None, layer_scale_conv=None, transformer_blocks=[]):
        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = [ConvBlock(dim=dim, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, layer_scale=layer_scale_conv) for i in range(depth)]
            self.transformer_block = False
        else:
            self.transformer_block = True
            self.blocks = [Block(dim=dim, counter=i, transformer_blocks=transformer_blocks, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, layer_scale=layer_scale) for i in range(depth)]
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def call(self, x):
        _, H, W, _ = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = tf.pad(x, [[0, 0], [0, pad_b], [0, pad_r], [0, 0]])
                Hp, Wp = H + pad_b, W + pad_r
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)

        for blk in self.blocks:
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :]

        if self.downsample is None:
            return x
        return self.downsample(x)


class MambaVision(Model):
    def __init__(self, dim, in_dim, depths, window_size, mlp_ratio, num_heads, drop_path_rate=0.2, in_chans=3, num_classes=1000, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., layer_scale=None, layer_scale_conv=None, **kwargs):
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.levels = []
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = MambaVisionLayer(dim=int(dim * 2 ** i), depth=depths[i], num_heads=num_heads[i], window_size=window_size[i], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, conv=conv, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], downsample=(i < 3), layer_scale=layer_scale, layer_scale_conv=layer_scale_conv, transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0 else list(range(depths[i] // 2, depths[i])), )
            self.levels.append(level)
        self.levels = tf.keras.Sequential(self.levels)
        self.norm = layers.BatchNormalization(epsilon=1e-5)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.head = layers.Dense(num_classes) if num_classes > 0 else layers.Lambda(lambda x: x)

    def call(self, x):
        x = self.patch_embed(x)
        x = self.levels(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = self.head(x)
        return x


def window_partition(x, window_size):
    _, H, W, C = x.shape
    x = tf.reshape(x, [-1, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, window_size * window_size, C])
    return x


def window_reverse(x, window_size, H, W):
    x = tf.reshape(x, [-1, H // window_size, W // window_size, window_size, window_size, -1])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, H, W, -1])
    return x





class Downsample(layers.Layer):
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = layers.Conv2D(dim_out, 3, strides=2, padding='same', use_bias=False)

    def call(self, x):
        x = self.reduction(x)
        return x


class PatchEmbed(layers.Layer):
    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = layers.Identity()
        self.conv_down = tf.keras.Sequential([
            layers.Conv2D(in_dim, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(epsilon=1e-4),
            layers.ReLU(),
            layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(epsilon=1e-4),
            layers.ReLU()
        ])

    def call(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(layers.Layer):
    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=3):
        super().__init__()
        self.conv1 = layers.Conv2D(dim, kernel_size=kernel_size, padding='same')
        self.norm1 = layers.BatchNormalization(epsilon=1e-5)
        self.act1 = layers.Activation('gelu')
        self.conv2 = layers.Conv2D(dim, kernel_size=kernel_size, padding='same')
        self.norm2 = layers.BatchNormalization(epsilon=1e-5)
        self.layer_scale = layer_scale

        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.gamma = self.add_weight(name='gamma', shape=(dim,), initializer=tf.keras.initializers.Constant(layer_scale), trainable=True)
            self.layer_scale = True
        else:
            self.layer_scale = False

        self.drop_path = layers.Dropout(drop_path) if drop_path > 0. else layers.Lambda(lambda x: x)

    def call(self, x):
        input_tensor = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * tf.reshape(self.gamma, (1, 1, 1, -1))
        x = input_tensor + self.drop_path(x)
        return x


class Attention(layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=layers.LayerNormalization):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.q_norm = norm_layer() if qk_norm else layers.Lambda(lambda x: x)
        self.k_norm = norm_layer() if qk_norm else layers.Lambda(lambda x: x)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = tf.reshape(self.qkv(x), (B, N, 3, self.num_heads, self.head_dim))
        q, k, v = tf.unstack(qkv, axis=2)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = tf.matmul(attn, v)

        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(layers.Layer):
    def __init__(self, dim, num_heads, counter, transformer_blocks, mlp_ratio=4., qkv_bias=False, qk_scale=False, drop=0., attn_drop=0., drop_path=0., act_layer=layers.Activation, norm_layer=layers.LayerNormalization, Mlp_block=layers.Dense, layer_scale=None):
        super().__init__()
        self.norm1 = norm_layer()
        if counter in transformer_blocks:
            self.mixer = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_scale, attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer)
        else:
            self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)

        self.drop_path = layers.Dropout(drop_path) if drop_path > 0. else layers.Lambda(lambda x: x)
        self.norm2 = norm_layer()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = tf.keras.Sequential([
            Mlp_block(mlp_hidden_dim, activation=act_layer),
            layers.Dropout(drop)
        ])
        self.gamma_1 = self.add_weight(name='gamma_1', shape=(dim,), initializer=tf.keras.initializers.Constant(layer_scale), trainable=True) if layer_scale is not None and isinstance(layer_scale, (int, float)) else 1
        self.gamma_2 = self.add_weight(name='gamma_2', shape=(dim,), initializer=tf.keras.initializers.Constant(layer_scale), trainable=True) if layer_scale is not None and isinstance(layer_scale, (int, float)) else 1

    def call(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(layers.Layer):
    def __init__(self, dim, depth, num_heads, window_size, conv=False, downsample=True, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., layer_scale=None, layer_scale_conv=None, transformer_blocks=[]):
        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = [ConvBlock(dim=dim, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, layer_scale=layer_scale_conv) for i in range(depth)]
            self.transformer_block = False
        else:
            self.transformer_block = True
            self.blocks = [Block(dim=dim, counter=i, transformer_blocks=transformer_blocks, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, layer_scale=layer_scale) for i in range(depth)]
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def call(self, x):
        _, H, W, _ = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = tf.pad(x, [[0, 0], [0, pad_b], [0, pad_r], [0, 0]])
                Hp, Wp = H + pad_b, W + pad_r
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)

        for blk in self.blocks:
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :]

        if self.downsample is None:
            return x
        return self.downsample(x)


class MambaVision(Model):
    def __init__(self, dim, in_dim, depths, window_size, mlp_ratio, num_heads, drop_path_rate=0.2, in_chans=3, num_classes=1000, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., layer_scale=None, layer_scale_conv=None, **kwargs):
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.levels = []
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = MambaVisionLayer(dim=int(dim * 2 ** i), depth=depths[i], num_heads=num_heads[i], window_size=window_size[i], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, conv=conv, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], downsample=(i < 3), layer_scale=layer_scale, layer_scale_conv=layer_scale_conv, transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0 else list(range(depths[i] // 2, depths[i])), )
            self.levels.append(level)
        self.levels = tf.keras.Sequential(self.levels)
        self.norm = layers.BatchNormalization(epsilon=1e-5)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.head = layers.Dense(num_classes) if num_classes > 0 else layers.Lambda(lambda x: x)

    def call(self, x):
        x = self.patch_embed(x)
        x = self.levels(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = self.head(x)
        return x


def window_partition(x, window_size):
    _, H, W, C = x.shape
    x = tf.reshape(x, [-1, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, window_size * window_size, C])
    return x


def window_reverse(x, window_size, H, W):
    x = tf.reshape(x, [-1, H // window_size, W // window_size, window_size, window_size, -1])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, H, W, -1])
    return x


