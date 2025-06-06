import sys
import types

def stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    def __getattr__(attr):
        if attr.startswith('pytest'):
            raise AttributeError
        return object
    mod.__getattr__ = __getattr__
    mod.__path__ = []
    try:
        from importlib.machinery import ModuleSpec
        mod.__spec__ = ModuleSpec(name, loader=None)
    except Exception:
        mod.__spec__ = None
    return mod


def setup_dependency_stubs() -> None:
    """Insert lightweight stubs for optional heavy dependencies if they are not installed."""
    heavy_modules = [
        'tensorflow', 'tensorflow.keras', 'tensorflow.keras.models',
        'tensorflow.keras.layers', 'tensorflow.keras.callbacks', 'tensorflow.keras.optimizers',
        'xgboost', 'lightgbm', 'optuna', 'backtrader',
        'river', 'faiss', 'faiss_cpu', 'shap',
        'matplotlib', 'matplotlib.pyplot', 'seaborn', 'textblob', 'boruta',
        'sklearn', 'numpy', 'pandas', 'requests', 'pycoingecko',
        'multipart', 'binance'
    ]

    for name in heavy_modules:
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except Exception:
            sys.modules[name] = stub_module(name)

    # Additional attributes for specific stubs
    if 'pandas' in sys.modules and 'Series' not in sys.modules['pandas'].__dict__:
        pd_stub = sys.modules['pandas']
        pd_stub.Series = type('Series', (), {'pct_change': lambda *a, **k: None})
        pd_stub.DataFrame = type('DataFrame', (), {'pct_change': lambda *a, **k: None})
        sys.modules.setdefault('pandas.api', stub_module('pandas.api'))
        sys.modules.setdefault('pandas.api.types', stub_module('pandas.api.types'))

    if 'tensorflow' in sys.modules:
        tf = sys.modules['tensorflow']
        tf.keras = sys.modules.setdefault('tensorflow.keras', stub_module('tensorflow.keras'))
        tf.keras.models = sys.modules.setdefault('tensorflow.keras.models', stub_module('tensorflow.keras.models'))
        tf.keras.layers = sys.modules.setdefault('tensorflow.keras.layers', stub_module('tensorflow.keras.layers'))
        tf.keras.callbacks = sys.modules.setdefault('tensorflow.keras.callbacks', stub_module('tensorflow.keras.callbacks'))
        tf.keras.optimizers = sys.modules.setdefault('tensorflow.keras.optimizers', stub_module('tensorflow.keras.optimizers'))
        tf.keras.models.Sequential = object
        tf.keras.models.Model = object
        tf.keras.models.load_model = lambda *a, **k: None
        tf.keras.layers.Dense = object
        tf.keras.layers.LSTM = object
        tf.keras.layers.GRU = object
        tf.keras.layers.Dropout = object
        tf.keras.layers.Input = object
        tf.keras.layers.Bidirectional = object
        tf.keras.layers.Concatenate = object
        tf.keras.layers.BatchNormalization = object
        tf.keras.callbacks.EarlyStopping = object
        tf.keras.callbacks.ReduceLROnPlateau = object
        tf.keras.callbacks.ModelCheckpoint = object
        tf.keras.optimizers.Adam = object
        tf.keras.callbacks.Callback = object

    if 'numpy' in sys.modules and getattr(sys.modules['numpy'], 'ndarray', None) is None:
        np_stub = sys.modules['numpy']
        np_stub.ndarray = object

    if 'sklearn' in sys.modules and 'metrics' not in sys.modules['sklearn'].__dict__:
        sklearn = sys.modules['sklearn']
        for sub in [
            'preprocessing', 'ensemble', 'linear_model', 'svm',
            'model_selection', 'metrics', 'neighbors', 'decomposition',
            'pipeline', 'feature_selection'
        ]:
            submod = types.ModuleType(f'sklearn.{sub}')
            submod.__getattr__ = lambda name, _default=object: object
            setattr(sklearn, sub, submod)
            sys.modules.setdefault(f'sklearn.{sub}', submod)
        sklearn.metrics.mean_squared_error = lambda *a, **k: 0
        sklearn.metrics.mean_absolute_error = lambda *a, **k: 0
        sklearn.metrics.r2_score = lambda *a, **k: 0

    if 'multipart' in sys.modules and '__version__' not in sys.modules['multipart'].__dict__:
        sys.modules['multipart'].__version__ = '0.0'
        sub_multipart = stub_module('multipart.multipart')
        sub_multipart.parse_options_header = lambda *a, **k: (b'multipart/form-data', {})
        sys.modules.setdefault('multipart.multipart', sub_multipart)

    if 'pycoingecko' in sys.modules:
        sys.modules['pycoingecko'].CoinGeckoAPI = object

    if 'matplotlib.pyplot' in sys.modules:
        pass  # stub already added via heavy_modules list
    else:
        sys.modules.setdefault('matplotlib.pyplot', stub_module('matplotlib.pyplot'))

    if 'binance' in sys.modules:
        binance_client = stub_module('binance.client')
        binance_client.Client = object
        sys.modules.setdefault('binance.client', binance_client)
        sys.modules['binance'].client = binance_client

    if 'backtrader' in sys.modules:
        backtrader_feeds = stub_module('backtrader.feeds')
        backtrader_feeds.PandasData = object
        sys.modules.setdefault('backtrader.feeds', backtrader_feeds)
        sys.modules['backtrader'].feeds = backtrader_feeds
