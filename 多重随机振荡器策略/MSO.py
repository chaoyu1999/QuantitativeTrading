import os
import random
import ccxt
import pandas as pd
import time
import smtplib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import aiosmtplib
from email.mime.text import MIMEText
from email.header import Header
import asyncio
from dotenv import load_dotenv
load_dotenv()  # 从.env文件加载环境变量

# 参数设置
_PROXY = True
Top_Symbols = 3 # 成交量排名前top的币

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 记录 subject 上次出现的时间
last_seen = {}
# 各个 timeframe 对应的分钟数
timeframe_minutes = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "1d": 1440,
    "1w": 10080
}


def can_use_subject(subject, timeframe):
    """判断当前 subject 是否可以使用"""
    now = time.time()

    # 如果 subject 之前没有记录，直接可以使用
    if subject not in last_seen:
        last_seen[subject] = now
        return True

    # 否则计算距离上次使用的时间差
    elapsed_minutes = (now - last_seen[subject]) / 60
    wait_minutes = timeframe_minutes[timeframe] * 5

    if elapsed_minutes >= wait_minutes:
        last_seen[subject] = now  # 更新记录时间
        return True
    else:
        # 还没到冷却时间，不允许使用
        return False

class MultiStochasticStrategy:
    """多重随机振荡器策略(url:https://cn.tradingview.com/v/qT7M70Aw/)"""

    def __init__(self, config_mode: str = "中线（平衡型）", custom_params: Dict = None):
        """
        初始化策略

        Args:
            config_mode: 配置模式 ["Custom", "短线（激进型）", "中线（平衡型）", "长线（稳健型）"]
            custom_params: 自定义参数字典
        """
        self.config_mode = config_mode
        self.custom_params = custom_params or {}
        self.extra_smooth = 2
        self.params = {}
        # 初始化参数
        self.setup_parameters()

    def setup_parameters(self):
        """根据配置模式设置参数"""
        if self.config_mode == "短线（激进型）":
            self.params = {
                'stoch1': {'length': 5, 'smoothK': 2, 'smoothD': 2},
                'stoch2': {'length': 9, 'smoothK': 1, 'smoothD': 2},
                'stoch3': {'length': 21, 'smoothK': 3, 'smoothD': 2},
                'stoch4': {'length': 34, 'smoothK': 5, 'smoothD': 1}
            }
        elif self.config_mode == "中线（平衡型）":
            self.params = {
                'stoch1': {'length': 7, 'smoothK': 3, 'smoothD': 3},
                'stoch2': {'length': 14, 'smoothK': 3, 'smoothD': 3},
                'stoch3': {'length': 21, 'smoothK': 5, 'smoothD': 3},
                'stoch4': {'length': 55, 'smoothK': 8, 'smoothD': 3}
            }
        elif self.config_mode == "长线（稳健型）":
            self.params = {
                'stoch1': {'length': 14, 'smoothK': 5, 'smoothD': 5},
                'stoch2': {'length': 21, 'smoothK': 5, 'smoothD': 5},
                'stoch3': {'length': 34, 'smoothK': 8, 'smoothD': 5},
                'stoch4': {'length': 89, 'smoothK': 13, 'smoothD': 5}
            }
        else:  # 自定义模式
            self.params = {
                'stoch1': {'length': self.custom_params.get('length1', 21),
                           'smoothK': self.custom_params.get('smoothK1', 3),
                           'smoothD': self.custom_params.get('smoothD1', 3)},
                'stoch2': {'length': self.custom_params.get('length2', 34),
                           'smoothK': self.custom_params.get('smoothK2', 3),
                           'smoothD': self.custom_params.get('smoothD2', 3)},
                'stoch3': {'length': self.custom_params.get('length3', 55),
                           'smoothK': self.custom_params.get('smoothK3', 5),
                           'smoothD': self.custom_params.get('smoothD3', 3)},
                'stoch4': {'length': self.custom_params.get('length4', 89),
                           'smoothK': self.custom_params.get('smoothK4', 10),
                           'smoothD': self.custom_params.get('smoothD4', 3)}
            }

    def ema(self, series: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均"""
        return series.ewm(span=period, adjust=False).mean()

    def sma(self, series: pd.Series, period: int) -> pd.Series:
        """计算简单移动平均"""
        return series.rolling(window=period).mean()

    def stochastic_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series,
                              length: int, smoothK: int, smoothD: int, extra_smooth: int) -> Tuple[
        pd.Series, pd.Series]:
        """
        计算平滑随机振荡器

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            length: 随机振荡器周期
            smoothK: K值平滑周期
            smoothD: D值平滑周期
            extra_smooth: 额外平滑周期

        Returns:
            Tuple: (K值序列, D值序列)
        """
        # 计算原始随机值
        lowest_low = low.rolling(window=length).min()
        highest_high = high.rolling(window=length).max()
        k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low)

        # 平滑处理
        k_smoothed = self.ema(k_raw, smoothK)
        d_smoothed = self.ema(k_smoothed, smoothD)

        # 额外平滑
        k_final = self.sma(k_smoothed, extra_smooth)
        d_final = self.sma(d_smoothed, extra_smooth)

        return k_final, d_final

    def calculate_all_stochastics(self, df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """
        计算所有随机振荡器指标

        Args:
            df: 包含OHLCV数据的DataFrame

        Returns:
            Dict: 包含所有K/D值的字典
        """
        results = {}

        for i, (key, params) in enumerate(self.params.items(), 1):
            k, d = self.stochastic_oscillator(
                df['high'], df['low'], df['close'],
                params['length'], params['smoothK'], params['smoothD'],
                self.extra_smooth
            )

            # 获取最新值
            k_value = k.iloc[-1] if not k.isna().iloc[-1] else None
            d_value = d.iloc[-1] if not d.isna().iloc[-1] else None

            results[f'k{i}'] = k_value
            results[f'd{i}'] = d_value
            results[f'{key}_k'] = k_value
            results[f'{key}_d'] = d_value

        return results

    def generate_signals(self, stoch_values: Dict) -> Dict[str, bool]:
        """
        生成交易信号

        Args:
            stoch_values: 随机振荡器数值字典

        Returns:
            Dict: 包含各种信号的字典
        """
        signals = {}

        # 提取K值
        k_values = [stoch_values[f'k{i}'] for i in range(1, 5) if stoch_values[f'k{i}'] is not None]
        d_values = [stoch_values[f'd{i}'] for i in range(1, 5) if stoch_values[f'd{i}'] is not None]

        if not k_values or not d_values:
            return signals

        # 超卖信号：所有K/D值都小于20
        signals['oversold'] = all(k < 20 for k in k_values) and all(d < 20 for d in d_values)

        # 超买信号：所有K/D值都大于80
        signals['overbought'] = all(k > 80 for k in k_values) and all(d > 80 for d in d_values)

        return signals





class TradingBot:
    """交易机器人主类"""

    def __init__(self, config_mode: str = "中线（平衡型）"):
        """
        初始化交易机器人

        Args:
            config_mode: 策略配置模式
        """
        self.strategy = MultiStochasticStrategy(config_mode)

        # 文件配置
        self.config_file = "symbols.txt"

        # ✅ 邮箱配置
        self.email_list = self.load_email_list("emails.txt")  # 从文件加载邮箱列表
        self.email_from = os.getenv("EMAIL_FROM")               # 无默认值，若缺失会返回None
        self.smtp_server = os.getenv("SMTP_SERVER")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))     # 带默认值 + 类型转换
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")                 # 敏感信息安全读取

        # 时间周期配置
        self.timeframes = list(timeframe_minutes.keys())

        # 邮件发送频率控制
        self.email_interval = 5 * 60
        self.last_email_sent = {}

        # 初始化交易所
        self.exchange = self.setup_exchange()

    # ===========================
    # 邮箱与交易所配置
    # ===========================
    def load_email_list(self, filename: str = "emails.txt") -> List[str]:
        """从文件读取收件人邮箱列表"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                emails = [line.strip() for line in f if line.strip()]
                logging.info(f"已从 {filename} 读取 {len(emails)} 个收件人邮箱")
                return emails
        except Exception as e:
            logging.warning(f"读取邮箱列表失败: {e}")
            return []

    def setup_exchange(self):
        """设置交易所连接"""
        if _PROXY:
            return ccxt.okx({
               'proxies': {
                   'http': 'http://127.0.0.1:60000',
                   'https': 'http://127.0.0.1:60000',
               },
                'options': {
                    'defaultType': 'swap',
                },
            })
        else:
            return ccxt.okx({
                'options': {
                    'defaultType': 'swap',
                },
            })
    # ===========================
    # 邮件发送部分
    # ===========================
    def send_email(self, subject: str, content: str) -> bool:
        """发送邮件给所有收件人"""
        if not self.email_list:
            logging.error("未找到收件人邮箱，邮件未发送。")
            return False

        max_retries = 100
        server = None

        for attempt in range(max_retries):
            try:
                msg = MIMEText(content, "plain", "utf-8")
                msg['From'] = Header(f"Trading Bot <{self.email_from}>")
                msg['To'] = Header("For you")  # 群发显示
                msg['Subject'] = Header(subject)

                logging.info(f"尝试发送邮件: {subject} (第 {attempt + 1}/{max_retries} 次)")

                server = smtplib.SMTP(self.smtp_server, timeout=30)
                server.ehlo()

                if server.has_extn('STARTTLS'):
                    server.starttls()
                    server.ehlo()

                server.login(self.smtp_user, self.smtp_pass)
                server.sendmail(self.email_from, self.email_list, msg.as_string())  # ✅ 群发
                server.quit()
                logging.info(f"邮件发送成功: {subject}")
                server.close()  # quit失败就强制关闭
                return True
            except Exception as e:
                logging.error(f"邮件发送失败 (第 {attempt + 1}/{max_retries} 次): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
            finally:
                # ✅ 确保无论成功失败都关闭连接
                if server is not None:
                    try:
                        server.quit()
                    except:
                        try:
                            server.close()  # quit失败就强制关闭
                        except:
                            pass
                server = None
        return False


    async def send_email_async(self, subject: str, content: str, max_retries: int = 3,
                               retry_delay: float = 5.0) -> bool:
        """异步发送邮件给所有收件人，包含重试机制"""

        import asyncio
        from async_timeout import timeout
        if not self.email_list:
            logging.error("未找到收件人邮箱，邮件未发送。")
            return False

        msg = MIMEText(content, "html", "utf-8")
        msg['From'] = Header(f"Trading Bot <{self.email_from}>")
        msg['To'] = Header("you")
        msg['Subject'] = Header(subject)

        for attempt in range(max_retries):
            try:
                # 使用超时控制，避免长时间等待
                async with timeout(30):
                    await aiosmtplib.send(
                        msg,
                        hostname=self.smtp_server,
                        port=self.smtp_port,
                        start_tls=True,
                        username=self.smtp_user,
                        password=self.smtp_pass,
                        timeout=30,
                        recipients=self.email_list
                    )
                logging.info(f"邮件发送成功: {subject}")
                return True

            except asyncio.TimeoutError:
                logging.warning(f"邮件发送超时，尝试重试 ({attempt + 1}/{max_retries})")
            except aiosmtplib.SMTPException as e:
                # 对于SMTP特定的错误，有些错误可能不需要重试
                if hasattr(e, 'code') and e.code in (421, 450, 451):
                    # 这些是临时性错误，可以重试
                    logging.warning(f"SMTP临时错误 [{e.code}]，尝试重试 ({attempt + 1}/{max_retries}): {e}")
                else:
                    # 永久性错误，直接返回失败
                    logging.error(f"SMTP永久性错误: {e}")
                    return False
            except Exception as e:
                # 处理连接重置、EOF等网络错误
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['eof', 'connection reset', 'connection closed']):
                    logging.warning(f"网络连接错误，尝试重试 ({attempt + 1}/{max_retries}): {e}")
                else:
                    # 其他未知错误，记录并返回失败
                    logging.error(f"未知错误: {e}")
                    return False

            # 如果不是最后一次尝试，等待后重试
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # 指数退避
                logging.info(f"等待 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)
            else:
                logging.error(f"邮件发送失败，已达到最大重试次数 {max_retries}")

        return False


    # ===========================
    # 数据与策略逻辑部分
    # ===========================
    def get_top_symbols(self, count: int = 50) -> List[str]:
        """获取交易量前 N 的币种"""
        try:
            markets = self.exchange.fetch_markets()
            three_months_ago = datetime.utcnow() - timedelta(days=30)
            filtered = []
            tickers = self.exchange.fetch_tickers()

            for m in markets:
                try:
                    if (m['active'] and m['quote'] == 'USDT' and
                            ('swap' in m['type'] or 'futures' in m['type'])):
                        listing_ts = int(m['info'].get('listTime', 0)) / 1000
                        listing_time = datetime.utcfromtimestamp(listing_ts)
                        if listing_time < three_months_ago:
                            symbol = m['symbol']
                            tick = tickers.get(symbol)
                            if tick and 'info' in tick and 'volCcy24h' in tick['info']:
                                base_volume = tick['average'] * float(tick['info']['volCcy24h'])
                                filtered.append((symbol, base_volume))
                except Exception:
                    continue

            filtered.sort(key=lambda x: x[1], reverse=True)
            top_symbols = [s[0] for s in filtered[:count]]
            logging.info(f"获取到 {len(top_symbols)} 个交易对")
            return top_symbols

        except Exception as e:
            logging.error(f"获取交易对列表失败: {e}")
            return []

    def get_custom_symbols(self) -> List[str]:
        """从本地 symbols.txt 读取自定义币种"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                symbols = [line.strip() for line in f if line.strip()]
                logging.info(f"从配置文件读取 {len(symbols)} 个自定义交易对")
                return symbols
        except Exception as e:
            logging.warning(f"读取自定义交易对文件失败: {e}")
            return []

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """获取 K 线数据（带限流防护与指数退避机制）"""
        max_retries = 5
        base_delay = 0.5  # 初始等待 0.5 秒

        for attempt in range(max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

                if not ohlcv or len(ohlcv) < 100:
                    logging.warning(f"{symbol} {timeframe} K线数据不足 ({len(ohlcv)}) 条")
                    return None

                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df

            except Exception as e:
                err = str(e)

                # 判断是否为 OKX 限频错误
                if "Too Many Requests" in err or "50011" in err:
                    # 计算退避时间：base_delay * 2^attempt + 随机扰动
                    delay = base_delay * (2 ** attempt) + random.uniform(0.2, 1.0)
                    logging.warning(
                        f"⚠️ {symbol} {timeframe} 请求过于频繁，{attempt+1}/{max_retries} 次重试，"
                        f"等待 {delay:.2f} 秒..."
                    )
                    time.sleep(delay)
                    continue  # 再试一次

                # 其他错误直接报出
                logging.error(f"❌ 获取 {symbol} {timeframe} K线失败: {e}")
                return None

        logging.error(f"🚫 多次重试仍失败: {symbol} {timeframe}")
        return None

    def analyze_symbol(self, symbol: str, timeframe: str) -> Dict:
        """分析单个交易对"""

        try:
            df = self.fetch_ohlcv(symbol, timeframe)
            if df is None:
                return {}

            stoch_values = self.strategy.calculate_all_stochastics(df)
            signals = self.strategy.generate_signals(stoch_values)

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'stoch_values': stoch_values,
                'signals': signals,
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logging.error(f"分析 {symbol} {timeframe} 时出错: {e}")
            return {}


    # ===========================
    # 主监控逻辑
    # ===========================
    async def run_monitoring(self):
        """运行监控循环"""
        try:
            symbols = list(set(self.get_top_symbols(Top_Symbols) + self.get_custom_symbols()))
            symbols.sort()
            logging.info(f"开始监控 {len(symbols)} 个交易对")

            while True:
                content = ""
                for symbol in symbols:
                    _symbol_content = ""
                    for timeframe in self.timeframes:
                        try:
                            analysis = self.analyze_symbol(symbol, timeframe)
                            if not analysis:
                                continue

                            signals = analysis['signals']
                            alert_signals = []
                            if signals.get('oversold'):
                                alert_signals.append(('OVERSOLD', '超卖信号'))
                            if signals.get('overbought'):
                                alert_signals.append(('OVERBOUGHT', '超买信号'))
                            for sig_key, sig_name in alert_signals:
                                subject = ""
                                if signals.get('oversold'):
                                    # 红色看多
                                    subject = f'<div style="font-weight: bold; color: red;">{symbol} {timeframe} - {sig_name}</div>\n'
                                if signals.get('overbought'):
                                    # 绿色看空
                                    subject = f'<div style="font-weight: bold; color: green;">{symbol} {timeframe} - {sig_name}</div>\n'
                                _symbol_content += subject
                        except Exception as e:
                            logging.error(f"处理 {symbol} {timeframe} 时出错: {e}")
                            continue
                    if _symbol_content != "":
                        content = content + _symbol_content + "<br>\n"
                if content != "":
                    # 异步发送
                    from datetime import datetime
                    # 获取当前时间
                    current_time = datetime.now()
                    # 格式化为字符串（年月日时分秒）
                    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    await self.send_email_async(time_str + "-订阅信息", content)
                logging.info("本轮监控完成，等待下一轮...")
                time.sleep(5*60)
        except KeyboardInterrupt:
            logging.info("用户中断程序")
        except Exception as e:
            logging.error(f"监控循环错误: {e}")


if __name__ == "__main__":
    # 创建交易机器人实例
    bot = TradingBot(config_mode="Custom")

    asyncio.run(bot.run_monitoring())

