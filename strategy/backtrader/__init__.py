from .ma import MAStrategyClass
from .mma import MMAStrategyClass,ETFMAStrategyClass
from .ama import AMAStrategyClass,ETFAMAStrategyClass
from .turtle import TurtleStrategyClass
from .macd import MACDStrategyClass
from .macd_kdj import MACDKDJStrategyClass
from .ma_rsi import MARSIStrategyClass
from .macd_sar import MACDSARStrategyClass
from .sar import SARStrategyClass
from .cta import CTAStrategyClass

get_strategy_cls = {
    'ma': MAStrategyClass,
    'mma': MMAStrategyClass,
    'ama': AMAStrategyClass,
    'turtle': TurtleStrategyClass,
    'macd': MACDStrategyClass,
    'macd_kdj': MACDKDJStrategyClass,
    'ma_rsi': MARSIStrategyClass,
    'macd_sar': MACDSARStrategyClass,
    'sar': SARStrategyClass,
    'cta': CTAStrategyClass,
    'etf_ma': ETFMAStrategyClass,
    'etf_ama': ETFAMAStrategyClass
}