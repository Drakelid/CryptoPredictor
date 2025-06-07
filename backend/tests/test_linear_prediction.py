import sys
import types


def stub_module(name):
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

# Stub heavy optional dependencies similar to test_main
tf = stub_module('tensorflow')
tf.keras = stub_module('tensorflow.keras')
tf.keras.models = stub_module('tensorflow.keras.models')
tf.keras.layers = stub_module('tensorflow.keras.layers')
tf.keras.callbacks = stub_module('tensorflow.keras.callbacks')
tf.keras.optimizers = stub_module('tensorflow.keras.optimizers')
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
tf.keras.layers.__getattr__ = lambda name: object
tf.keras.callbacks.__getattr__ = lambda name: object
sys.modules.setdefault('tensorflow', tf)
sys.modules.setdefault('tensorflow.keras', tf.keras)
sys.modules.setdefault('tensorflow.keras.models', tf.keras.models)
sys.modules.setdefault('tensorflow.keras.layers', tf.keras.layers)
sys.modules.setdefault('tensorflow.keras.callbacks', tf.keras.callbacks)
sys.modules.setdefault('tensorflow.keras.optimizers', tf.keras.optimizers)

for module_name in [
    'xgboost', 'lightgbm', 'optuna', 'backtrader',
    'river', 'faiss', 'faiss_cpu', 'shap',
    'matplotlib', 'seaborn', 'schedule', 'textblob', 'boruta',
    'sklearn', 'pandas', 'pycoingecko'
]:
    sys.modules.setdefault(module_name, stub_module(module_name))
sys.modules.setdefault('requests', stub_module('requests'))

sys.modules.setdefault('multipart', stub_module('multipart'))
sys.modules['multipart'].__version__ = '0.0'
sub_multipart = stub_module('multipart.multipart')
sub_multipart.parse_options_header = lambda *a, **k: (b'multipart/form-data', {})
sys.modules.setdefault('multipart.multipart', sub_multipart)

sys.modules.setdefault('matplotlib.pyplot', stub_module('matplotlib.pyplot'))

sys.modules['pycoingecko'].CoinGeckoAPI = object
sys.modules.setdefault('binance', stub_module('binance'))
binance_client = stub_module('binance.client')
binance_client.Client = object
sys.modules.setdefault('binance.client', binance_client)
backtrader_feeds = stub_module('backtrader.feeds')
backtrader_feeds.PandasData = object
sys.modules.setdefault('backtrader.feeds', backtrader_feeds)
sys.modules['backtrader'].feeds = backtrader_feeds

# Minimal sklearn stub with common submodules
sklearn = sys.modules.setdefault('sklearn', types.ModuleType('sklearn'))
for sub in [
    'preprocessing', 'ensemble', 'linear_model', 'svm',
    'model_selection', 'metrics', 'neighbors', 'decomposition',
    'pipeline', 'feature_selection'
]:
    submod = types.ModuleType(f'sklearn.{sub}')
    submod.__getattr__ = lambda name, _default=object: object
    setattr(sklearn, sub, submod)
    sys.modules.setdefault(f'sklearn.{sub}', submod)

pd = sys.modules.setdefault('pandas', stub_module('pandas'))
pd.Series = type('Series', (), {'pct_change': lambda *a, **k: None})
pd.DataFrame = type('DataFrame', (), {'pct_change': lambda *a, **k: None})
sys.modules.setdefault('pandas.api', stub_module('pandas.api'))
sys.modules.setdefault('pandas.api.types', stub_module('pandas.api.types'))

sklearn_metrics = sys.modules.setdefault('sklearn.metrics', types.ModuleType('sklearn.metrics'))
sklearn_metrics.mean_squared_error = lambda *a, **k: 0
sklearn_metrics.mean_absolute_error = lambda *a, **k: 0
sklearn_metrics.r2_score = lambda *a, **k: 0

from datetime import datetime, timedelta

from app.services.prediction_service import PredictionService
from app.schemas.prediction import PredictionResult

class DummyDataService:
    def load_data(self, symbol: str, source: str = None):
        now = datetime(2021, 1, 1)
        timestamps = [now + timedelta(days=i) for i in range(10)]
        prices = [10 + i for i in range(10)]
        return {'timestamp': timestamps, 'close': prices}

class DummySentimentService:
    def get_sentiment_for_prediction(self, symbol: str):
        return {'social_sentiment': 0.6}

class DummyTrainingService:
    def get_model_info(self, *a, **k):
        return [types.SimpleNamespace(model_path='dummy')]

prediction_service = PredictionService(
    data_service=DummyDataService(),
    training_service=DummyTrainingService(),
    sentiment_service=DummySentimentService()
)

prediction_service._save_prediction = lambda *a, **k: None
from app.utils.prediction_feedback import prediction_feedback_system
prediction_feedback_system.store_prediction = lambda *a, **k: None


def test_linear_prediction():
    result = prediction_service._predict_linear('BTC', horizon=3, confidence_interval=True)
    assert isinstance(result, PredictionResult)
    assert len(result.values) == 3
