"""
Investment Masters Evaluation Module.

15 investment masters analyze market conditions through their unique philosophical lenses,
each producing a signal (bullish/bearish/neutral) with confidence and reasoning.
A portfolio manager synthesizes all opinions into a consensus recommendation.

Architecture:
1. Rule-based scoring: Each master has 3-5 criteria weighted to produce [-1, +1] score
2. LLM reasoning: Rules produce signal+confidence, LLM generates personalized reasoning
3. News overlay: Living masters' real opinions scraped and used to adjust confidence
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Signal constants ──────────────────────────────────────────────────
BULLISH = "bullish"
BEARISH = "bearish"
NEUTRAL = "neutral"

SIGNAL_ZH = {BULLISH: "看涨", BEARISH: "看空", NEUTRAL: "中性"}
SIGNAL_EMOJI = {BULLISH: "🟢", BEARISH: "🔴", NEUTRAL: "🟡"}


# ── Data Models ───────────────────────────────────────────────────────

@dataclass
class MasterSignal:
    master_id: str
    master_name_en: str
    master_name_zh: str
    signal: str  # bullish / bearish / neutral
    confidence: int  # 0-100
    reasoning: str = ""
    style_tags: list[str] = field(default_factory=list)
    philosophy_zh: str = ""
    famous_quote_zh: str = ""
    avatar_initials: str = ""
    is_alive: bool = False
    news_sentiment: Optional[str] = None
    news_headline: Optional[str] = None
    key_metrics: dict = field(default_factory=dict)
    last_updated: str = ""

    def to_dict(self) -> dict:
        return {
            "master_id": self.master_id,
            "master_name_en": self.master_name_en,
            "master_name_zh": self.master_name_zh,
            "signal": self.signal,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "style_tags": self.style_tags,
            "philosophy_zh": self.philosophy_zh,
            "famous_quote_zh": self.famous_quote_zh,
            "avatar_initials": self.avatar_initials,
            "is_alive": self.is_alive,
            "news_sentiment": self.news_sentiment,
            "news_headline": self.news_headline,
            "key_metrics": self.key_metrics,
            "last_updated": self.last_updated,
        }


@dataclass
class PortfolioDecision:
    overall_signal: str
    overall_confidence: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    consensus_score: float  # -100 to +100
    summary_zh: str
    recommended_action: str
    suggested_position: str
    top_bull_reasons: list[str] = field(default_factory=list)
    top_bear_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_signal": self.overall_signal,
            "overall_confidence": self.overall_confidence,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "consensus_score": round(self.consensus_score, 1),
            "summary_zh": self.summary_zh,
            "recommended_action": self.recommended_action,
            "suggested_position": self.suggested_position,
            "top_bull_reasons": self.top_bull_reasons,
            "top_bear_reasons": self.top_bear_reasons,
        }


# ── Master Profiles ───────────────────────────────────────────────────

MASTER_PROFILES = {
    "warren_buffett": {
        "name_en": "Warren Buffett",
        "name_zh": "沃伦·巴菲特",
        "is_alive": True,
        "avatar_initials": "WB",
        "style_tags": ["价值投资", "安全边际", "长期持有"],
        "philosophy_zh": "以合理价格买入优秀企业，长期持有。别人贪婪时恐惧，别人恐惧时贪婪。坚守能力圈，不投不懂的东西。",
        "famous_quote_zh": "别人贪婪时恐惧，别人恐惧时贪婪",
        "news_query": "Warren Buffett market outlook 2026",
    },
    "ben_graham": {
        "name_en": "Ben Graham",
        "name_zh": "本·格雷厄姆",
        "is_alive": False,
        "avatar_initials": "BG",
        "style_tags": ["深度价值", "安全边际", "烟蒂投资"],
        "philosophy_zh": "价值投资之父。强调安全边际，以远低于内在价值的价格买入。市场先生是仆人不是向导，利用他的情绪波动而非被他左右。",
        "famous_quote_zh": "市场短期是投票机，长期是称重机",
        "news_query": None,
    },
    "charlie_munger": {
        "name_en": "Charlie Munger",
        "name_zh": "查理·芒格",
        "is_alive": False,
        "avatar_initials": "CM",
        "style_tags": ["质量投资", "多元思维", "逆向思考"],
        "philosophy_zh": "以合理价格买入卓越企业胜过以低价买入普通企业。强调多元思维模型，用跨学科知识做投资决策。极度厌恶投机和杠杆。",
        "famous_quote_zh": "以合理价格买入卓越企业，胜过以低价买入平庸企业",
        "news_query": None,
    },
    "michael_burry": {
        "name_en": "Michael Burry",
        "name_zh": "迈克尔·布里",
        "is_alive": True,
        "avatar_initials": "MB",
        "style_tags": ["逆向投资", "做空泡沫", "深度基本面"],
        "philosophy_zh": "擅长发现市场泡沫和系统性风险。通过深入分析数据发现别人忽视的危险。2008年次贷危机前做空MBS一战成名。",
        "famous_quote_zh": "风险来自于你不知道自己在做什么",
        "news_query": "Michael Burry market warning 2026",
    },
    "bill_ackman": {
        "name_en": "Bill Ackman",
        "name_zh": "比尔·阿克曼",
        "is_alive": True,
        "avatar_initials": "BA",
        "style_tags": ["激进投资", "集中持仓", "质量复合增长"],
        "philosophy_zh": "集中投资少数高质量公司，必要时推动管理层变革。偏好具有持久竞争优势和自由现金流生成能力的企业。",
        "famous_quote_zh": "最好的投资是买入并持有伟大的公司",
        "news_query": "Bill Ackman market view 2026",
    },
    "cathie_wood": {
        "name_en": "Cathie Wood",
        "name_zh": "凯茜·伍德",
        "is_alive": True,
        "avatar_initials": "CW",
        "style_tags": ["颠覆性创新", "成长股", "长期视野"],
        "philosophy_zh": "专注于颠覆性创新技术，如AI、基因测序、区块链、机器人。愿意为高成长支付高估值，着眼5-10年长期回报。",
        "famous_quote_zh": "创新是经济增长的唯一驱动力",
        "news_query": "Cathie Wood AI outlook 2026",
    },
    "peter_lynch": {
        "name_en": "Peter Lynch",
        "name_zh": "彼得·林奇",
        "is_alive": False,
        "avatar_initials": "PL",
        "style_tags": ["成长价值", "PEG估值", "买你知道的"],
        "philosophy_zh": "投资你了解的公司。寻找PEG<1的成长股，即市盈率低于增长率的公司。散户可以通过日常观察发现十倍股。",
        "famous_quote_zh": "买你了解的公司",
        "news_query": None,
    },
    "phil_fisher": {
        "name_en": "Phil Fisher",
        "name_zh": "菲利普·费雪",
        "is_alive": False,
        "avatar_initials": "PF",
        "style_tags": ["成长股", "闲聊法", "长期持有"],
        "philosophy_zh": "通过闲聊法（scuttlebutt）深入研究公司管理层、研发能力和行业地位。买入真正有成长潜力的公司并长期持有。",
        "famous_quote_zh": "买入优秀公司然后坐着不动",
        "news_query": None,
    },
    "mohnish_pabrai": {
        "name_en": "Mohnish Pabrai",
        "name_zh": "莫尼什·帕布莱",
        "is_alive": True,
        "avatar_initials": "MP",
        "style_tags": ["Dhandho投资", "低风险高不确定", "克隆巴菲特"],
        "philosophy_zh": "Dhandho框架：低风险高不确定性的投资。押注、然后等待。极少交易，高集中度。克隆巴菲特和芒格的投资组合。",
        "famous_quote_zh": "低风险、高不确定性 = 好投资",
        "news_query": "Mohnish Pabrai market view 2026",
    },
    "nassim_taleb": {
        "name_en": "Nassim Taleb",
        "name_zh": "纳西姆·塔勒布",
        "is_alive": True,
        "avatar_initials": "NT",
        "style_tags": ["尾部风险", "反脆弱", "杠铃策略"],
        "philosophy_zh": "关注黑天鹅事件和尾部风险。市场低估了极端事件的概率和冲击。推荐杠铃策略：大部分资金安全+小部分购买远端保护。",
        "famous_quote_zh": "脆弱性的反面不是强韧，而是反脆弱",
        "news_query": "Nassim Taleb bubble risk 2026",
    },
    "aswath_damodaran": {
        "name_en": "Aswath Damodaran",
        "name_zh": "阿斯沃斯·达莫达兰",
        "is_alive": True,
        "avatar_initials": "AD",
        "style_tags": ["估值 Dean", "DCF", "叙事与数字"],
        "philosophy_zh": "估值Dean。每个资产都有内在价值，通过DCF和基本面数据可以估算。叙事驱动价格偏离价值，但最终会回归。",
        "famous_quote_zh": "价格是你要付的，价值是你要得的",
        "news_query": "Damodaran market valuation 2026",
    },
    "stanley_druckenmiller": {
        "name_en": "Stanley Druckenmiller",
        "name_zh": "斯坦利·德鲁肯米勒",
        "is_alive": True,
        "avatar_initials": "SD",
        "style_tags": ["宏观交易", "流动性驱动", "趋势跟随"],
        "philosophy_zh": "宏观经济驱动的顶级交易员。关注流动性和央行政策。趋势明确时重仓出击，犯错时迅速认错离场。",
        "famous_quote_zh": "重要的不是对错，而是对了赚多少、错了亏多少",
        "news_query": "Druckenmiller market outlook 2026",
    },
    "ray_dalio": {
        "name_en": "Ray Dalio",
        "name_zh": "瑞·达利欧",
        "is_alive": True,
        "avatar_initials": "RD",
        "style_tags": ["全天候", "经济机器", "债务周期"],
        "philosophy_zh": "理解经济如何像机器一样运转。债务周期决定市场方向。全天候策略：分散配置应对不同经济环境。",
        "famous_quote_zh": "痛苦 + 反思 = 进步",
        "news_query": "Ray Dalio economic cycle 2026",
    },
    "rakesh_jhunjhunwala": {
        "name_en": "Rakesh Jhunjhunwala",
        "name_zh": "拉克什·金君瓦拉",
        "is_alive": False,
        "avatar_initials": "RJ",
        "style_tags": ["新兴市场", "成长+价值", "印度巴菲特"],
        "philosophy_zh": "印度的巴菲特。看好新兴市场和国内消费故事。结合成长和价值，在合理价格买入增长型公司。",
        "famous_quote_zh": "保持乐观，印度会增长",
        "news_query": None,
    },
}


# ── Master Evaluator Base ─────────────────────────────────────────────

class Criterion:
    """Single evaluation criterion with weight and value."""

    def __init__(self, weight: float, value: float, description: str = ""):
        self.weight = weight
        self.value = max(-1.0, min(1.0, value))  # Clamp to [-1, +1]
        self.description = description


def compute_signal(criteria: list[Criterion]) -> tuple[str, int]:
    """
    Compute signal from weighted criteria.
    Returns (signal, confidence).
    """
    total = sum(c.weight * c.value for c in criteria)
    total = max(-1.0, min(1.0, total))

    if total > 0.3:
        signal = BULLISH
        confidence = min(90, int(50 + total * 40))
    elif total < -0.3:
        signal = BEARISH
        confidence = min(90, int(50 + abs(total) * 40))
    else:
        signal = NEUTRAL
        confidence = max(20, int(30 + (1 - abs(total)) * 20))

    return signal, confidence


def safe_float(val, default=None):
    """Safely convert to float."""
    try:
        if val is None or val == "" or val == "N/A":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


# ── Individual Master Evaluators ──────────────────────────────────────

def evaluate_buffett(data: dict) -> MasterSignal:
    """Warren Buffett: Value, margin of safety, contrarian fear/greed."""
    bubble_score = safe_float(data.get("bubble_score"), 50)
    cape = safe_float(data.get("cape"), 30)
    vix = safe_float(data.get("vix"), 20)
    spy_pe = safe_float(data.get("spy_pe"), 25)
    buffett_indicator = safe_float(data.get("buffett_indicator"), 150)

    criteria = [
        Criterion(0.25, -1.0 if bubble_score > 60 else (0.3 if bubble_score < 30 else 0), "泡沫评分"),
        Criterion(0.20, -1.0 if cape > 30 else (0.3 if cape < 20 else 0), "CAPE估值"),
        Criterion(0.15, -1.0 if spy_pe > 25 else (0.3 if spy_pe < 15 else 0), "SPY市盈率"),
        Criterion(0.20, 0.8 if vix > 25 else (-0.5 if vix < 12 else 0), "VIX逆向"),
        Criterion(0.20, -1.0 if buffett_indicator > 180 else (0.2 if buffett_indicator < 100 else 0), "巴菲特指标"),
    ]
    signal, confidence = compute_signal(criteria)

    return MasterSignal(
        master_id="warren_buffett",
        signal=signal, confidence=confidence,
        key_metrics={"bubble_score": bubble_score, "cape": cape, "vix": vix, "spy_pe": spy_pe},
        **_profile("warren_buffett"),
    )


def evaluate_graham(data: dict) -> MasterSignal:
    """Ben Graham: Deep value, margin of safety, Mr. Market."""
    spy_pe = safe_float(data.get("spy_pe"), 25)
    cape = safe_float(data.get("cape"), 30)
    bubble_score = safe_float(data.get("bubble_score"), 50)
    vix = safe_float(data.get("vix"), 20)

    criteria = [
        Criterion(0.35, -1.0 if spy_pe > 20 else (0.5 if spy_pe < 12 else 0), "SPY PE>20严重看空"),
        Criterion(0.30, -1.0 if cape > 25 else (0.3 if cape < 18 else 0), "CAPE>25看空"),
        Criterion(0.20, -0.8 if bubble_score > 50 else 0.3, "泡沫评分"),
        Criterion(0.15, 0.7 if vix > 30 else (-0.3 if vix < 15 else 0), "VIX>30逆向买入"),
    ]
    signal, confidence = compute_signal(criteria)

    return MasterSignal(
        master_id="ben_graham",
        signal=signal, confidence=confidence,
        key_metrics={"spy_pe": spy_pe, "cape": cape, "vix": vix},
        **_profile("ben_graham"),
    )


def evaluate_munger(data: dict) -> MasterSignal:
    """Charlie Munger: Quality at fair price, anti-speculation."""
    bubble_score = safe_float(data.get("bubble_score"), 50)
    concentration = safe_float(data.get("concentration"), 15)
    vix = safe_float(data.get("vix"), 20)
    qqq_rsi = safe_float(data.get("qqq_rsi"), 50)

    criteria = [
        Criterion(0.30, -1.0 if bubble_score > 50 else 0.2, "泡沫>50投机狂热"),
        Criterion(0.25, -0.8 if concentration > 18 else 0.2, "集中度>18%危险"),
        Criterion(0.25, -0.6 if qqq_rsi > 70 else 0.2, "RSI>70过热"),
        Criterion(0.20, 0.5 if vix > 25 else 0, "市场恐慌时考虑买入"),
    ]
    signal, confidence = compute_signal(criteria)

    return MasterSignal(
        master_id="charlie_munger",
        signal=signal, confidence=confidence,
        key_metrics={"bubble_score": bubble_score, "concentration": concentration, "qqq_rsi": qqq_rsi},
        **_profile("charlie_munger"),
    )


def evaluate_burry(data: dict) -> MasterSignal:
    """Michael Burry: Contrarian short, bubble hunter."""
    bubble_score = safe_float(data.get("bubble_score"), 50)
    vix = safe_float(data.get("vix"), 20)
    spy_change = safe_float(data.get("spy_change"), 0)
    concentration = safe_float(data.get("concentration"), 15)

    # Burry is naturally bearish-biased
    vix_complacency = -0.8 if vix < 15 else (0.5 if vix > 30 else 0)
    bubble_signal = -1.0 if bubble_score > 50 else (-0.5 if bubble_score > 30 else 0)

    criteria = [
        Criterion(0.30, bubble_signal, "泡沫>50强烈看空"),
        Criterion(0.25, vix_complacency, "VIX低位+泡沫高=极度危险"),
        Criterion(0.25, -0.6 if concentration > 15 else 0, "集中度高风险大"),
        Criterion(0.20, -0.5 if spy_change > 30 else 0, "涨幅过大警惕回调"),
    ]
    signal, confidence = compute_signal(criteria)

    return MasterSignal(
        master_id="michael_burry",
        signal=signal, confidence=confidence,
        key_metrics={"bubble_score": bubble_score, "vix": vix, "concentration": concentration},
        **_profile("michael_burry"),
    )


def evaluate_ackman(data: dict) -> MasterSignal:
    """Bill Ackman: Quality compounders, concentrated bets."""
    nvda_change = safe_float(data.get("nvda_change"), 0)
    bubble_score = safe_float(data.get("bubble_score"), 50)
    spy_change = safe_float(data.get("spy_change"), 0)
    vix = safe_float(data.get("vix"), 20)

    criteria = [
        Criterion(0.30, 0.7 if nvda_change > 20 else (-0.5 if nvda_change < -20 else 0.2), "质量股增长"),
        Criterion(0.25, -0.7 if bubble_score > 60 else (0.2 if bubble_score < 30 else 0), "泡沫>60减仓"),
        Criterion(0.25, 0.5 if spy_change > 0 and vix < 20 else (-0.3 if spy_change < -10 else 0), "趋势+低VIX"),
        Criterion(0.20, 0.3, "偏好高质量持仓"),
    ]
    signal, confidence = compute_signal(criteria)

    return MasterSignal(
        master_id="bill_ackman",
        signal=signal, confidence=confidence,
        key_metrics={"nvda_change": nvda_change, "bubble_score": bubble_score, "spy_change": spy_change},
        **_profile("bill_ackman"),
    )


def evaluate_wood(data: dict) -> MasterSignal:
    """Cathie Wood: Disruptive innovation, growth at any price."""
    nvda_change = safe_float(data.get("nvda_change"), 0)
    qqq_rsi = safe_float(data.get("qqq_rsi"), 50)
    qqq_change = safe_float(data.get("qqq_change"), 0)
    bubble_score = safe_float(data.get("bubble_score"), 50)

    criteria = [
        Criterion(0.40, 0.8 if nvda_change > 30 else (0.3 if nvda_change > 0 else -0.5), "NVDA增速"),
        Criterion(0.20, 0.5 if qqq_change > 0 else -0.3, "QQQ趋势"),
        Criterion(0.20, 0.3 if qqq_rsi < 80 else -0.5, "RSI不过热则看多"),
        Criterion(0.20, -0.4 if bubble_score > 70 else 0.2, "仅极度泡沫时谨慎"),
    ]
    signal, confidence = compute_signal(criteria)

    return MasterSignal(
        master_id="cathie_wood",
        signal=signal, confidence=confidence,
        key_metrics={"nvda_change": nvda_change, "qqq_rsi": qqq_rsi, "qqq_change": qqq_change},
        **_profile("cathie_wood"),
    )


def evaluate_lynch(data: dict) -> MasterSignal:
    """Peter Lynch: PEG, growth at reasonable price, buy what you know."""
    nvda_change = safe_float(data.get("nvda_change"), 0)
    qqq_rsi = safe_float(data.get("qqq_rsi"), 50)
    spy_pe = safe_float(data.get("spy_pe"), 25)
    bubble_score = safe_float(data.get("bubble_score"), 50)

    # PEG approximation: if growth (nvda_change) > PE, PEG < 1 (good)
    peg_proxy = safe_float(spy_pe / max(nvda_change, 1), 2.0)

    criteria = [
        Criterion(0.35, 0.7 if peg_proxy < 1.5 else (-0.5 if peg_proxy > 3 else 0), "PEG估值匹配度"),
        Criterion(0.25, -0.6 if qqq_rsi > 70 else 0.2, "RSI>70市场跑太快"),
        Criterion(0.20, -0.5 if bubble_score > 50 else 0.2, "泡沫过高谨慎"),
        Criterion(0.20, 0.5 if nvda_change > 20 else 0, "增长真实则看多"),
    ]
    signal, confidence = compute_signal(criteria)

    return MasterSignal(
        master_id="peter_lynch",
        signal=signal, confidence=confidence,
        key_metrics={"peg_proxy": peg_proxy, "qqq_rsi": qqq_rsi, "nvda_change": nvda_change},
        **_profile("peter_lynch"),
    )


def evaluate_fisher(data: dict) -> MasterSignal:
    """Phil Fisher: Scuttlebutt, growth companies, long-term hold."""
    nvda_change = safe_float(data.get("nvda_change"), 0)
    cape = safe_float(data.get("cape"), 30)
    bubble_score = safe_float(data.get("bubble_score"), 50)
    spy_change = safe_float(data.get("spy_change"), 0)

    criteria = [
        Criterion(0.40, 0.7 if nvda_change > 30 else (0.2 if nvda_change > 0 else -0.4), "科技增长质量"),
        Criterion(0.30, -0.8 if cape > 35 else (0.2 if cape < 25 else 0), "CAPE>35好公司也太贵"),
        Criterion(0.15, -0.4 if bubble_score > 60 else 0, "高泡沫谨慎"),
        Criterion(0.15, 0.3 if spy_change > 0 else 0, "趋势向上加分"),
    ]
    signal, confidence = compute_signal(criteria)

    return MasterSignal(
        master_id="phil_fisher",
        signal=signal, confidence=confidence,
        key_metrics={"nvda_change": nvda_change, "cape": cape, "bubble_score": bubble_score},
        **_profile("phil_fisher"),
    )


def evaluate_pabrai(data: dict) -> MasterSignal:
    """Mohnish Pabrai: Dhandho, low risk high uncertainty, clones Buffett."""
    bubble_score = safe_float(data.get("bubble_score"), 50)
    cape = safe_float(data.get("cape"), 30)
    vix = safe_float(data.get("vix"), 20)
    spy_pe = safe_float(data.get("spy_pe"), 25)

    criteria = [
        Criterion(0.35, -0.8 if bubble_score > 60 else (0.3 if bubble_score < 25 else 0), "泡沫>60低安全边际"),
        Criterion(0.25, -0.7 if cape > 30 else (0.3 if cape < 20 else 0), "CAPE估值"),
        Criterion(0.20, -0.5 if spy_pe > 25 else 0.3, "PE过高无安全边际"),
        Criterion(0.20, 0.5 if vix > 25 else 0, "VIX高=机会"),
    ]
    signal, confidence = compute_signal(criteria)

    return MasterSignal(
        master_id="mohnish_pabrai",
        signal=signal, confidence=confidence,
        key_metrics={"bubble_score": bubble_score, "cape": cape, "spy_pe": spy_pe},
        **_profile("mohnish_pabrai"),
    )


def evaluate_taleb(data: dict) -> MasterSignal:
    """Nassim Taleb: Tail risk, antifragility, barbell strategy."""
    vix = safe_float(data.get("vix"), 20)
    bubble_score = safe_float(data.get("bubble_score"), 50)
    concentration = safe_float(data.get("concentration"), 15)
    spy_change = safe_float(data.get("spy_change"), 0)

    criteria = [
        Criterion(0.30, -1.0 if vix < 15 else (0.5 if vix > 30 else -0.3), "VIX<15尾部风险被忽视"),
        Criterion(0.25, -0.8 if bubble_score > 40 else 0, "泡沫>40=脆弱系统"),
        Criterion(0.25, -0.7 if concentration > 15 else 0, "集中度>15%=系统依赖少数"),
        Criterion(0.20, -0.5 if spy_change > 20 else 0, "涨幅大=更脆弱"),
    ]
    signal, confidence = compute_signal(criteria)

    return MasterSignal(
        master_id="nassim_taleb",
        signal=signal, confidence=confidence,
        key_metrics={"vix": vix, "bubble_score": bubble_score, "concentration": concentration},
        **_profile("nassim_taleb"),
    )


def evaluate_damodaran(data: dict) -> MasterSignal:
    """Aswath Damodaran: Valuation Dean, DCF, narrative vs numbers."""
    cape = safe_float(data.get("cape"), 30)
    bubble_score = safe_float(data.get("bubble_score"), 50)
    spy_pe = safe_float(data.get("spy_pe"), 25)

    # ERP approximation: lower ERP = more overvalued
    # If PE is high, implied earnings yield (1/PE) is low vs risk-free rate
    erp_proxy = (1.0 / max(spy_pe, 1)) * 100  # Earnings yield %
    erp_signal = -0.8 if erp_proxy < 3.5 else (0.3 if erp_proxy > 5 else 0)

    criteria = [
        Criterion(0.30, -1.0 if cape > 30 else (0.3 if cape < 20 else 0), "CAPE估值"),
        Criterion(0.25, erp_signal, "股权风险溢价"),
        Criterion(0.25, -0.7 if bubble_score > 50 else 0.2, "泡沫评分"),
        Criterion(0.20, -0.5 if spy_pe > 30 else (0.2 if spy_pe < 15 else 0), "PE水平"),
    ]
    signal, confidence = compute_signal(criteria)

    return MasterSignal(
        master_id="aswath_damodaran",
        signal=signal, confidence=confidence,
        key_metrics={"cape": cape, "erp_proxy": round(erp_proxy, 2), "spy_pe": spy_pe},
        **_profile("aswath_damodaran"),
    )


def evaluate_druckenmiller(data: dict) -> MasterSignal:
    """Stanley Druckenmiller: Macro, liquidity, trend follower."""
    qqq_rsi = safe_float(data.get("qqq_rsi"), 50)
    spy_change = safe_float(data.get("spy_change"), 0)
    vix = safe_float(data.get("vix"), 20)
    bubble_score = safe_float(data.get("bubble_score"), 50)

    criteria = [
        Criterion(0.30, 0.7 if spy_change > 0 else -0.5, "趋势跟随"),
        Criterion(0.25, 0.5 if qqq_rsi < 70 and qqq_rsi > 40 else (-0.5 if qqq_rsi > 80 else 0), "RSI趋势区间"),
        Criterion(0.25, -0.8 if vix > 30 else (0.3 if vix < 20 else 0), "VIX急升快速退出"),
        Criterion(0.20, -0.4 if bubble_score > 70 else 0.2, "极端泡沫减仓"),
    ]
    signal, confidence = compute_signal(criteria)

    return MasterSignal(
        master_id="stanley_druckenmiller",
        signal=signal, confidence=confidence,
        key_metrics={"spy_change": spy_change, "qqq_rsi": qqq_rsi, "vix": vix},
        **_profile("stanley_druckenmiller"),
    )


def evaluate_dalio(data: dict) -> MasterSignal:
    """Ray Dalio: All Weather, economic machine, debt cycles."""
    bubble_score = safe_float(data.get("bubble_score"), 50)
    vix = safe_float(data.get("vix"), 20)
    spy_change = safe_float(data.get("spy_change"), 0)
    cape = safe_float(data.get("cape"), 30)

    # Dalio is balanced: weighs risks vs opportunities
    criteria = [
        Criterion(0.30, -0.6 if bubble_score > 60 else (0.3 if bubble_score < 30 else 0), "债务周期位置"),
        Criterion(0.25, -0.5 if vix > 25 else (0.3 if vix < 15 else 0), "市场压力指标"),
        Criterion(0.25, -0.4 if cape > 30 else 0.2, "长期估值回归"),
        Criterion(0.20, 0.3 if abs(spy_change) < 10 else (-0.3 if spy_change > 20 else 0), "波动与相关性"),
    ]
    signal, confidence = compute_signal(criteria)

    return MasterSignal(
        master_id="ray_dalio",
        signal=signal, confidence=confidence,
        key_metrics={"bubble_score": bubble_score, "vix": vix, "cape": cape},
        **_profile("ray_dalio"),
    )


def evaluate_jhunjhunwala(data: dict) -> MasterSignal:
    """Rakesh Jhunjhunwala: Emerging markets, growth + value hybrid."""
    spy_pe = safe_float(data.get("spy_pe"), 25)
    bubble_score = safe_float(data.get("bubble_score"), 50)
    qqq_change = safe_float(data.get("qqq_change"), 0)
    # A-share proxy: use bubble score inverse for emerging market opportunity
    a_share_pe_proxy = safe_float(30 - bubble_score * 0.2, 20)  # Approximation

    criteria = [
        Criterion(0.30, 0.6 if spy_pe < 20 else (-0.4 if spy_pe > 30 else 0), "PE<20看多"),
        Criterion(0.25, -0.5 if bubble_score > 50 else 0.2, "全球泡沫>50减仓"),
        Criterion(0.25, 0.4 if qqq_change > 0 else -0.2, "新兴市场增长"),
        Criterion(0.20, 0.3 if a_share_pe_proxy < 25 else -0.2, "A股估值"),
    ]
    signal, confidence = compute_signal(criteria)

    return MasterSignal(
        master_id="rakesh_jhunjhunwala",
        signal=signal, confidence=confidence,
        key_metrics={"spy_pe": spy_pe, "bubble_score": bubble_score, "a_share_pe_proxy": a_share_pe_proxy},
        **_profile("rakesh_jhunjhunwala"),
    )


def _profile(master_id: str) -> dict:
    """Extract profile fields from MASTER_PROFILES."""
    p = MASTER_PROFILES.get(master_id, {})
    return {
        "master_name_en": p.get("name_en", master_id),
        "master_name_zh": p.get("name_zh", master_id),
        "is_alive": p.get("is_alive", False),
        "avatar_initials": p.get("avatar_initials", "?"),
        "style_tags": p.get("style_tags", []),
        "philosophy_zh": p.get("philosophy_zh", ""),
        "famous_quote_zh": p.get("famous_quote_zh", ""),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


# ── Master Registry ───────────────────────────────────────────────────

# Ordered list of (evaluator_func, master_id)
MASTER_EVALUATORS = [
    (evaluate_buffett, "warren_buffett"),
    (evaluate_graham, "ben_graham"),
    (evaluate_munger, "charlie_munger"),
    (evaluate_burry, "michael_burry"),
    (evaluate_ackman, "bill_ackman"),
    (evaluate_wood, "cathie_wood"),
    (evaluate_lynch, "peter_lynch"),
    (evaluate_fisher, "phil_fisher"),
    (evaluate_pabrai, "mohnish_pabrai"),
    (evaluate_taleb, "nassim_taleb"),
    (evaluate_damodaran, "aswath_damodaran"),
    (evaluate_druckenmiller, "stanley_druckenmiller"),
    (evaluate_dalio, "ray_dalio"),
    (evaluate_jhunjhunwala, "rakesh_jhunjhunwala"),
]


# ── Main Engine ───────────────────────────────────────────────────────

class InvestmentMastersEngine:
    """
    Orchestrates the evaluation of all investment masters.

    1. Runs rule-based evaluation for each master
    2. Optionally enriches with news sentiment (living masters)
    3. Generates LLM reasoning (if available)
    4. Returns all signals for portfolio manager synthesis
    """

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, use_llm: bool = False):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._scanner = None
        # LLM is disabled by default to avoid slow API responses.
        # Set USE_LLM_FOR_MASTERS=true in .env to enable.
        self._llm_available = bool(self.api_key) and use_llm
        self._llm_enabled = os.environ.get("USE_LLM_FOR_MASTERS", "").lower() == "true"

    def evaluate(self, market_data: dict) -> dict:
        """
        Run all master evaluations and return structured result.

        Args:
            market_data: Dict with keys like bubble_score, cape, vix, spy_pe,
                        nvda_change, qqq_rsi, qqq_change, spy_change, etc.

        Returns:
            Dict with "masters" list and "portfolio_decision".
        """
        signals = []

        # Step 1: Rule-based evaluation
        for evaluator, master_id in MASTER_EVALUATORS:
            try:
                signal = evaluator(market_data)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Error evaluating {master_id}: {e}")
                signals.append(self._fallback_signal(master_id))

        # Step 2: News enrichment SKIPPED — HTTP scraping is too slow and unreliable.
        # News sentiment was designed to adjust confidence, but rule-based signals are sufficient.
        # if self._scanner is None: ...  (commented out for performance)

        # Step 3: LLM reasoning generation (or fallback to template)
        if self._llm_available:
            try:
                reasonings = self._generate_batch_reasoning(signals, market_data)
                for s, r in zip(signals, reasonings):
                    s.reasoning = r
            except Exception as e:
                logger.error(f"Batch LLM reasoning failed: {e}")
                for s in signals:
                    s.reasoning = self._template_reasoning(s)
        else:
            # No LLM available, use template reasoning
            for s in signals:
                s.reasoning = self._template_reasoning(s)

        # Step 4: Portfolio manager synthesis
        portfolio_decision = PortfolioManagerSynthesizer().synthesize(signals)

        return {
            "masters": [s.to_dict() for s in signals],
            "portfolio_decision": portfolio_decision.to_dict(),
            "last_scanned": datetime.now(timezone.utc).isoformat(),
        }

    def _fallback_signal(self, master_id: str) -> MasterSignal:
        """Fallback signal when evaluation fails."""
        p = _profile(master_id)
        return MasterSignal(
            master_id=master_id,
            signal=NEUTRAL, confidence=20,
            reasoning="评估失败，使用默认中性信号",
            **p,
        )

    def _generate_batch_reasoning(self, signals: list[MasterSignal], market_data: dict) -> list[str]:
        """Generate reasoning for ALL masters in batched LLM calls (2 batches of 7)."""
        BATCH_SIZE = 7
        all_reasonings = []

        for batch_start in range(0, len(signals), BATCH_SIZE):
            batch = signals[batch_start:batch_start + BATCH_SIZE]
            master_inputs = []
            for s in batch:
                profile = MASTER_PROFILES.get(s.master_id, {})
                metrics_str = ", ".join(f"{k}={v}" for k, v in s.key_metrics.items() if v is not None)
                sig_zh = SIGNAL_ZH.get(s.signal, '中性')
                master_inputs.append(
                    f"- {s.master_name_zh}: {profile.get('philosophy_zh', '')[:20]} | "
                    f"数据={metrics_str[:60]} | 判断={sig_zh}({s.confidence}%)"
                )

            prompt = (
                f"以下是{len(batch)}位投资大师的市场分析结果。请分别用一句话（25字以内）"
                f"给出每位大师的投资观点，解释该判断。严格按JSON数组格式输出：\n\n"
                + "\n".join(master_inputs)
            )

            try:
                from openai import OpenAI
                import re
                client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=500,
                    timeout=30,
                )
                text = response.choices[0].message.content.strip()

                # Parse JSON array
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    batch_reasonings = json.loads(match.group())
                else:
                    batch_reasonings = [
                        re.sub(r'^[\d\-\*•\[\]]+\s*', '', line).strip()
                        for line in text.split('\n') if line.strip()
                    ]

                while len(batch_reasonings) < len(batch):
                    batch_reasonings.append(self._template_reasoning(batch[len(batch_reasonings)]))
                all_reasonings.extend([r[:80] for r in batch_reasonings[:len(batch)]])

            except Exception as e:
                logger.debug(f"Batch LLM failed (batch {batch_start//BATCH_SIZE + 1}): {e}")
                all_reasonings.extend([self._template_reasoning(s) for s in batch])

        return all_reasonings

    def _generate_reasoning(self, signal: MasterSignal, market_data: dict) -> str:
        """Generate personalized reasoning using LLM. Deprecated: use _generate_batch_reasoning instead."""
        return self._template_reasoning(signal)

    def _template_reasoning(self, signal: MasterSignal) -> str:
        """Generate master-specific reasoning when LLM is unavailable."""
        metrics = signal.key_metrics
        signal_zh = SIGNAL_ZH.get(signal.signal, "中性")

        if signal.master_id == "warren_buffett":
            bs = metrics.get("bubble_score", 50)
            vix = metrics.get("vix", 20)
            if signal.signal == BEARISH:
                return f"泡沫评分{bs}，估值远超安全边际。我1969年解散合伙基金就是因为找不到便宜的股票——历史押着相同的韵脚。"
            elif signal.signal == BULLISH:
                return f"VIX {vix}，恐慌情绪充足。别人恐惧时贪婪，这正是以好价格买入伟大企业的机会。"
            return f"估值偏高但未到极端，保持耐心。好机会需要等待，现金不是垃圾，是一种选择权。"

        elif signal.master_id == "ben_graham":
            pe = metrics.get("spy_pe", 25)
            vix = metrics.get("vix", 20)
            if signal.signal == BEARISH:
                return f"SPY市盈率{pe}倍远超安全边际。市场先生又在他的躁狂期报价了——聪明的投资者应该卖出而非买入。"
            return f"估值缺乏足够安全边际。按照防御型投资者标准，PE>15就不合格。宁可错过不做错。"

        elif signal.master_id == "charlie_munger":
            bs = metrics.get("bubble_score", 50)
            qqq_rsi = metrics.get("qqq_rsi", 50)
            if signal.signal == BEARISH:
                return f"泡沫{bs}、RSI {qqq_rsi}——从众效应驱动的价格上涨。反过来想：什么情况会让这笔投资失败？答案太多。"
            return f"市场尚未出现明显的质量折扣。以合理价格买入卓越企业——当前两个条件都没满足，等待。"

        elif signal.master_id == "michael_burry":
            bs = metrics.get("bubble_score", 50)
            vix = metrics.get("vix", 20)
            if signal.signal == BEARISH:
                return f"泡沫{bs}，VIX {vix}。数据不说谎——尾部风险定价不足是系统性危机的前兆。2008年的剧本正在重演。"
            return f"数据尚未显示迫在眉睫的系统性风险，但信贷扩张和杠杆水平仍需密切监控。保持警惕。"

        elif signal.master_id == "bill_ackman":
            nvda = metrics.get("nvda_change", 0)
            bs = metrics.get("bubble_score", 50)
            if signal.signal == BULLISH:
                return f"质量股增长强劲(NVDA {nvda}%)。我宁愿集中持有3-5只高确信度的股票，而不是分散在20只平庸的标的上。"
            return f"泡沫{bs}，安全边际不足。在估值过高时，现金不是垃圾——它是一种选择权。"

        elif signal.master_id == "cathie_wood":
            nvda = metrics.get("nvda_change", 0)
            if signal.signal == BULLISH:
                return f"AI创新周期远未结束(NVDA增长{nvda}%)。颠覆性技术遵循S型曲线，早期增长看似线性实为指数。传统PE估值完全错失了这一点。"
            return f"创新叙事出现短期裂痕，但这不改变5-10年的长期趋势。忽略噪音，关注技术采用曲线。"

        elif signal.master_id == "peter_lynch":
            peg = metrics.get("peg_proxy", 2)
            if signal.signal == BULLISH:
                return f"PEG≈{peg:.1f}，增长与估值匹配度尚可。我在麦哲伦的经验：买你了解的公司，从日常生活中发现机会。"
            return f"PEG {peg:.1f}太高，市场增速跟不上估值。好公司不等于好价格——这是我最重要的教训。"

        elif signal.master_id == "phil_fisher":
            nvda = metrics.get("nvda_change", 0)
            if signal.signal == BULLISH:
                return f"科技龙头增长质量优秀(NVDA {nvda}%)。通过闲聊法验证：供应商、客户、竞争对手的反馈都支持长期增长叙事。"
            return f"即使是最好的公司，当价格严重偏离内在价值时也不值得持有。耐心是成长股投资者最好的朋友。"

        elif signal.master_id == "mohnish_pabrai":
            bs = metrics.get("bubble_score", 50)
            if signal.signal == BEARISH:
                return f"泡沫{bs}，缺乏安全边际。Dhandho的核心是'低风险高不确定性'——当前环境下两者都不具备。什么都不做。"
            return f"估值尚可但未出现'太容易拒绝'的赌注。99%的时间你应该什么都不做，等待赔率和概率同时站在你这边。"

        elif signal.master_id == "nassim_taleb":
            vix = metrics.get("vix", 20)
            bs = metrics.get("bubble_score", 50)
            if signal.signal == BEARISH:
                return f"VIX {vix}，泡沫{bs}。尾部风险被严重低估——市场假设正态分布，现实是肥尾的。建议杠铃策略：90%安全+10%远端保护。"
            return f"系统性脆弱性暂未达到临界点，但永远不要用'平衡配置'替代杠铃策略。中间地带的资产在危机中跌得最惨。"

        elif signal.master_id == "aswath_damodaran":
            cape = metrics.get("cape", 30)
            pe = metrics.get("spy_pe", 25)
            if signal.signal == BEARISH:
                return f"CAPE {cape}倍，PE {pe}倍。DCF模型显示当前价格隐含的增长假设过于乐观。叙事必须回归基本面，数字不会说谎。"
            return f"估值虽不便宜但也没到泡沫程度。关键是：你买入的资产质量能否支撑这个价格？用FCFF和WACC验证。"

        elif signal.master_id == "stanley_druckenmiller":
            spy = metrics.get("spy_change", 0)
            vix = metrics.get("vix", 20)
            if signal.signal == BULLISH:
                return f"趋势仍然向上(SPY {spy:+.0f}%)，流动性环境有支撑。我犯过的最大错误就是无视趋势而坚持自己的观点。顺势而为。"
            return f"趋势转弱，VIX {vix}。当赔率不再有利时，快速认错离场比坚持正确更重要。"

        elif signal.master_id == "ray_dalio":
            bs = metrics.get("bubble_score", 50)
            if signal.signal == BEARISH:
                return f"泡沫{bs}，债务周期进入后期阶段。历史表明，每一次泡沫都以'这次不一样'开始，以去杠杆收场。做好分散配置。"
            return f"经济机器仍在运转，但需关注周期位置。全天候策略的核心是：不预测、不择时、在任何经济环境下都有资产表现良好。"

        elif signal.master_id == "rakesh_jhunjhunwala":
            bs = metrics.get("bubble_score", 50)
            if signal.signal == BULLISH:
                return f"估值合理，新兴市场的增长故事没有改变。中产阶级扩大和消费升级是全球最大的投资机会。保持乐观。"
            return f"全球泡沫{bs}，新兴市场也难以独善其身。但悲观者看起来聪明，乐观者赚钱——长期来看增长会解决大部分问题。"

        return f"基于当前数据分析，维持{signal_zh}立场，置信度{signal.confidence}%。"


# ── Portfolio Manager Synthesizer ─────────────────────────────────────

class PortfolioManagerSynthesizer:
    """
    Synthesizes all master opinions into a consensus recommendation.
    """

    def synthesize(self, signals: list[MasterSignal]) -> PortfolioDecision:
        bullish = [s for s in signals if s.signal == BULLISH]
        bearish = [s for s in signals if s.signal == BEARISH]
        neutral = [s for s in signals if s.signal == NEUTRAL]

        total = len(signals)
        if total == 0:
            return self._neutral_decision()

        # Consensus score: +100 for all bullish, -100 for all bearish
        score_map = {BULLISH: 1, BEARISH: -1, NEUTRAL: 0}
        weighted_sum = sum(s.confidence * score_map.get(s.signal, 0) for s in signals)
        max_possible = sum(s.confidence for s in signals)
        consensus_score = (weighted_sum / max_possible * 100) if max_possible > 0 else 0

        # Overall signal
        if consensus_score > 20:
            overall = BULLISH
        elif consensus_score < -20:
            overall = BEARISH
        else:
            overall = NEUTRAL

        # Overall confidence: average of all confidences
        overall_confidence = int(sum(s.confidence for s in signals) / total)

        # Recommended action
        action, position = self._recommend_action(overall, consensus_score, len(bullish), len(bearish), total)

        # Summary
        summary = self._generate_summary(bullish, bearish, neutral, overall, consensus_score)

        # Top reasons
        top_bull = [s.reasoning for s in bullish[:3]]
        top_bear = [s.reasoning for s in bearish[:3]]

        return PortfolioDecision(
            overall_signal=overall,
            overall_confidence=overall_confidence,
            bullish_count=len(bullish),
            bearish_count=len(bearish),
            neutral_count=len(neutral),
            consensus_score=consensus_score,
            summary_zh=summary,
            recommended_action=action,
            suggested_position=position,
            top_bull_reasons=top_bull,
            top_bear_reasons=top_bear,
        )

    def _recommend_action(self, signal: str, score: float, bull: int, bear: int, total: int) -> tuple[str, str]:
        if signal == BEARISH and score < -50:
            return "大幅减仓", "50% 现金, 30% 对冲, 20% 持仓"
        elif signal == BEARISH:
            return "减仓防守", "30% 现金, 40% 对冲, 30% 持仓"
        elif signal == BULLISH and score > 50:
            return "积极做多", "10% 现金, 10% 对冲, 80% 持仓"
        elif signal == BULLISH:
            return "适度做多", "20% 现金, 20% 对冲, 60% 持仓"
        else:
            return "保持观望", "40% 现金, 30% 对冲, 30% 持仓"

    def _generate_summary(self, bullish, bearish, neutral, overall, score) -> str:
        total = len(bullish) + len(bearish) + len(neutral)
        direction = "看多" if overall == BULLISH else ("看空" if overall == BEARISH else "中性")

        summary = f"{total}位投资大师中，{len(bullish)}位{SIGNAL_ZH[BULLISH]}，{len(bearish)}位{SIGNAL_ZH[BEARISH]}，{len(neutral)}位{SIGNAL_ZH[NEUTRAL]}。"
        summary += f"综合共识偏向{direction}（得分{score:+.1f}）。"

        if bullish:
            names = ", ".join(s.master_name_zh for s in bullish[:2])
            summary += f"{names}等认为"
            summary += bullish[0].reasoning[:30] + "；" if bullish[0].reasoning else ""

        if bearish:
            names = ", ".join(s.master_name_zh for s in bearish[:2])
            summary += f"{names}等认为"
            summary += bearish[0].reasoning[:30] + "。" if bearish[0].reasoning else ""

        return summary

    def _neutral_decision(self) -> PortfolioDecision:
        return PortfolioDecision(
            overall_signal=NEUTRAL,
            overall_confidence=0,
            bullish_count=0,
            bearish_count=0,
            neutral_count=0,
            consensus_score=0,
            summary_zh="暂无大师评估数据",
            recommended_action="保持观望",
            suggested_position="50% 现金, 25% 对冲, 25% 持仓",
        )


# ── Convenience Function ──────────────────────────────────────────────

def build_masters_evaluation(
    bubble_score: float = 50,
    cape: float = 30,
    vix: float = 20,
    spy_pe: float = 25,
    spy_change: float = 0,
    qqq_change: float = 0,
    qqq_rsi: float = 50,
    nvda_change: float = 0,
    concentration: float = 15,
    buffett_indicator: float = 150,
    api_key: str = None,
    use_llm: bool = False,
) -> dict:
    """
    Convenience function to run the full masters evaluation.

    This is the entry point called from web_terminal.py's get_all_data().

    Args:
        use_llm: Set True to enable LLM reasoning generation. Default False
                 because LLM calls add significant latency. When False,
                 template-based reasoning is used instead.
    """
    market_data = {
        "bubble_score": bubble_score,
        "cape": cape,
        "vix": vix,
        "spy_pe": spy_pe,
        "spy_change": spy_change,
        "qqq_change": qqq_change,
        "qqq_rsi": qqq_rsi,
        "nvda_change": nvda_change,
        "concentration": concentration,
        "buffett_indicator": buffett_indicator,
    }

    engine = InvestmentMastersEngine(api_key=api_key, use_llm=use_llm)
    return engine.evaluate(market_data)


# ── Master Profile Generation (for hover popup) ──────────────────────

# Static track records for each master: 3-4 famous historical trades
MASTER_TRACK_RECORDS = {
    "warren_buffett": [
        {"year": "1988", "title": "买入可口可乐", "detail": "耗资10亿美元买入可口可乐，至今回报超15倍，成为伯克希尔最经典持仓"},
        {"year": "2016", "title": "建仓苹果公司", "detail": "逐步建仓Apple，到2024年持仓市值超1800亿，占组合40%以上"},
        {"year": "2008", "title": "高盛救赎交易", "detail": "金融危机期间向高盛注资50亿美元优先股，获10%股息+认股权证"},
        {"year": "1972", "title": "收购See's Candies", "detail": "2500万美元收购，此后50年创造超20亿利润，教会了'为品质支付合理价'"},
    ],
    "ben_graham": [
        {"year": "1940s", "title": "发现GEICO", "detail": "发现政府员工保险公司的烟蒂价值，为伯克希尔带来超2300倍回报"},
        {"year": "1929", "title": "大萧条后的重生", "detail": "1929年崩盘几乎破产后，总结出安全边际和内在价值投资理论"},
        {"year": "1934", "title": "《证券分析》出版", "detail": "奠定价值投资理论基础，定义了'内在价值'和'安全边际'概念"},
    ],
    "charlie_munger": [
        {"year": "1960s", "title": "Wheek&Munger基金", "detail": "17年复合年化19.8%，远超同期道指表现"},
        {"year": "1972", "title": "推动收购See's Candies", "detail": "说服巴菲特以溢价收购优质企业，改变了伯克希尔的投资风格"},
        {"year": "2020s", "title": "Daily Journal投资", "detail": "在90多岁高龄仍重仓阿里巴巴，展现对中国消费力的信心"},
    ],
    "michael_burry": [
        {"year": "2005-2007", "title": "次贷做空一战成名", "detail": "通过分析房贷数据发现次级贷款泡沫，通过CDS做空获利数十亿，《大空头》原型"},
        {"year": "2020", "title": "精准做空科技股", "detail": "预警科技股泡沫，大幅做空ARKK等高估值成长基金"},
        {"year": "2021", "title": "SPAC泡沫预警", "detail": "公开警告SPAC狂热，其基金提前退出大量SPAC空头头寸"},
    ],
    "bill_ackman": [
        {"year": "2012", "title": "做空康宝莱", "detail": "公开做空 Herbalife，最终获利数亿，展现了激进投资者的影响力"},
        {"year": "2020 COVID", "title": "CDS对冲完美交易", "detail": "花2700万买入CDS对冲，疫情期间价值飙至26亿，99倍回报"},
        {"year": "2014-2015", "title": "建仓 Valeant", "detail": "重仓 Valeant Pharmaceuticals，后因公司丑闻亏损27%并公开认错"},
    ],
    "cathie_wood": [
        {"year": "2020", "title": "ARKK翻倍神话", "detail": "ARK Innovation ETF全年涨幅150%，押注特斯拉、Zoom、Roku等大获成功"},
        {"year": "2017-2020", "title": "重仓特斯拉", "detail": "在特斯拉70-400美元期间持续买入，2021年目标价喊到3000美元"},
        {"year": "2021", "title": "Coinbase上市", "detail": "提前布局加密货币经济，Coinbase上市当日ARK持仓浮盈超10亿"},
    ],
    "peter_lynch": [
        {"year": "1977-1990", "title": "麦哲伦基金传奇", "detail": "管理麦哲伦基金13年，复合年化29%，1800美元变14万7千美元"},
        {"year": "1980s", "title": "发现十倍股", "detail": "在麦哲伦期间发现了Dunkin' Donuts、Home Depot等数十只十倍股"},
        {"year": "1990", "title": "急流勇退", "detail": "在基金巅峰时选择退休，认为'知道何时离开和知道如何投资一样重要'"},
    ],
    "phil_fisher": [
        {"year": "1955", "title": "买入摩托罗拉", "detail": "1955年买入摩托罗拉，持有超过30年，回报超过100倍"},
        {"year": "1958", "title": "《普通股与不普通利润》", "detail": "定义了'闲聊法'(Scuttlebutt)和成长股投资15要点"},
        {"year": "1990s", "title": "持续持有优质股", "detail": "90多岁高龄仍保持对科技股的深刻理解，持续为巴菲特提供灵感"},
    ],
    "mohnish_pabrai": [
        {"year": "1999", "title": "Pabrai投资基金", "detail": "创立投资基金，25年复合年化约15%，采用极低换手率策略"},
        {"year": "2021", "title": "抄底比亚迪", "detail": "克隆巴菲特逻辑，公开表示比亚迪是'下一个特斯拉'"},
        {"year": "2008", "title": "拍卖跟投巴菲特", "detail": "以65万美元拍下与巴菲特午餐机会，此后多次成功复制巴菲特持仓"},
    ],
    "nassim_taleb": [
        {"year": "1987", "title": "黑色星期一天鹅", "detail": "1987年崩盘中获利，因持有远端看跌期权，验证了尾部风险管理理论"},
        {"year": "2000s", "title": "Empirica Capital", "detail": "运营专注于尾部风险的对冲基金，在平稳年份小亏、危机年份大赚"},
        {"year": "2006", "title": "《黑天鹅》出版", "detail": "预见了2007年金融危机，书中描述的脆弱性在次贷危机中一一验证"},
    ],
    "aswath_damodaran": [
        {"year": "2000", "title": "科技泡沫估值警告", "detail": "在2000年泡沫高峰发表多篇论文警告科技股估值离谱，用数据证明"},
        {"year": "2008", "title": "危机中精确估值", "detail": "在金融危机中对银行进行DCF估值，精确判断底部买入时机"},
        {"year": "2021", "title": "Tesla估值争议", "detail": "公开质疑Tesla 1万亿美元估值，认为'叙事已完全脱离基本面'"},
    ],
    "stanley_druckenmiller": [
        {"year": "1992", "title": "做空英镑", "detail": "与索罗斯合作做空英镑，单笔交易获利超10亿，成为金融史经典"},
        {"year": "2020", "title": "抗疫交易", "detail": "3月初清仓转空，3月底满仓杀回，抓住了V型反弹的转折点"},
        {"year": "2023", "title": "AI趋势重仓", "detail": "公开表示'AI是改变游戏规则的'，将大部分仓位集中于AI相关标的"},
    ],
    "ray_dalio": [
        {"year": "1987", "title": "黑色星期一教训", "detail": "1987年预判错误亏损60%后，发明了全天候风险平价策略"},
        {"year": "2008", "title": "全天候策略表现", "detail": "Pure Alpha基金在2008年获得+9%正收益，市场恐慌中的避风港"},
        {"year": "2010s", "title": "预测债务周期", "detail": "准确预测了2011年债务上限危机和欧债危机的演变路径"},
    ],
    "rakesh_jhunjhunwala": [
        {"year": "2003", "title": "泰坦公司大赚", "detail": "3元买入Titan Company，持有至200元以上，获利超过60倍"},
        {"year": "2000s", "title": "印度股市教父", "detail": "从零起步到管理50亿美元组合，被称为'印度的巴菲特'"},
        {"year": "2022", "title": "最后建仓", "detail": "去世前重仓印度航空和消费股，展现对印度增长故事的最终信心"},
    ],
}


# ── Persona-Specific LLM System Prompts ───────────────────────────────
# Each master has a unique analysis framework, voice, and decision logic
# modeled after the ai-hedge-fund agent definitions.

MASTER_LLM_SYSTEM_PROMPTS = {
    "warren_buffett": {
        "framework": """你是沃伦·巴菲特。你的分析必须围绕以下框架：
1. **护城河分析**：当前市场中最强的企业护城河是什么？品牌、网络效应、转换成本？
2. **所有者收益（Owner Earnings）**：不要只看净利润，看自由现金流 - 资本支出
3. **DCF估值**：以无风险利率折现未来现金流，当前价格是否提供了足够的内在价值折扣？
4. **安全边际**：只有在价格远低于内在价值时才买入
5. **市场先生**：利用市场情绪而非被其左右

你的表达风格：用简单语言和类比解释复杂概念，引用可口可乐、苹果等实际案例，语气平和但观点鲜明。避免技术术语和花哨词汇。""",
    },
    "ben_graham": {
        "framework": """你是本杰明·格雷厄姆，价值投资之父。你的分析必须围绕以下框架：
1. **安全边际计算**：以净资产价值、清算价值为锚，当前价格相对于NNAV的折扣
2. **市场先生比喻**：市场是躁郁的生意伙伴，利用它的价格波动而非听从它的建议
3. **防御型投资者标准**：PE < 15，PB < 1.5，债务/权益 < 0.5，连续10年分红
4. **进取型投资者标准**：当前价格 < 营运资本净值的2/3
5. **内在价值估算**：基于EPS、分红、资产价值的保守估算

你的表达风格：学术严谨、数据驱动、保守审慎。用具体数字说话，给出明确的估值区间。对投机行为直言不讳地批评。""",
    },
    "charlie_munger": {
        "framework": """你是查理·芒格。你的分析必须围绕以下框架：
1. **多元思维模型**：用心理学、物理学、生物学、数学的跨学科原理分析市场
2. **逆向思考（Inversion）**：要明白什么会导致投资失败，然后避开它
3. **护城河强度**：品牌、规模效应、网络效应、转换成本、专利
4. **能力圈**：只在理解的领域内投资，不碰看不懂的科技或商业模式
5. **人类误判心理学**：激励机制、从众效应、锚定偏见、损失厌恶

你的表达风格：睿智、幽默、犀利。常用'让我告诉你'、'这是常识'等口吻。引用历史故事和跨学科类比。对愚蠢行为毫不留情地讽刺。""",
    },
    "michael_burry": {
        "framework": """你是迈克尔·布里，Scion Asset Management创始人。你的分析必须围绕以下框架：
1. **数据驱动的异常检测**：从底层数据发现市场忽视的风险——信贷数据、杠杆水平、流动性指标
2. **FCF收益率 & EV/EBIT**：自由现金流收益率是真实回报的最佳指标，EV/EBIT比PE更可靠
3. **资产负债表安全性**：高杠杆是危机的催化剂，关注企业债务/EBITDA比率
4. **内部人交易信号**：高管增持是最可靠的看涨信号，减持反之
5. **尾部风险定位**：市场系统性低估尾部事件概率，VIX长期偏低本身就是风险

你的表达风格：简洁、直接、数据导向。不说废话，用数字和事实证明观点。语气偏冷峻、有时带有危机预警的紧迫感。""",
    },
    "bill_ackman": {
        "framework": """你是比尔·阿克曼，Pershing Square创始人。你的分析必须围绕以下框架：
1. **质量复合增长（Quality Compounders）**：寻找具有持久竞争优势、高ROIC、可预测现金流的公司
2. **集中度策略**：集中投资8-12只高确信度股票，而非广泛分散
3. **自由现金流生成能力**：关注企业产生自由现金流的能力，这是价值的核心
4. ** activist潜力**：管理层是否有改进空间？能否通过推动变革释放价值？
5. **下行保护**：在追求高质量的同时确保有足够的下行保护

你的表达风格：自信、直接、分析性强。喜欢用'让我解释一下为什么'这样的引导语。对管理层质量有独到见解。""",
    },
    "cathie_wood": {
        "framework": """你是凯茜·伍德，ARK Invest创始人。你的分析必须围绕以下框架：
1. **颠覆性创新识别**：AI、基因编辑、自动驾驶、区块链、机器人——哪些技术正在改变范式？
2. **指数级增长曲线**：创新技术遵循S型采用曲线，早期增长看似线性实则指数
3. **TAM分析（总可寻址市场）**：不要看当前收入，看10年后的市场空间
4. **研发密度**：高研发投入的公司是未来赢家，这不是费用是投资
5. **成本曲线下降**：技术成本按学习曲线下降，摩尔定律式进步

你的表达风格：乐观、前瞻性、充满激情。喜欢用'改变游戏规则'、'范式转变'、'指数级'等词汇。对传统估值方法嗤之以鼻。""",
    },
    "peter_lynch": {
        "framework": """你是彼得·林奇，麦哲伦基金传奇经理。你的分析必须围绕以下框架：
1. **PEG估值**：市盈率低于增长率的公司是被低估的成长股，PEG < 1是买入信号
2. **六种公司分类**：缓慢增长、稳健增长、快速增长、周期型、困境反转、资产隐蔽
3. **买你知道的**：从日常生活观察中发现投资机会——商场、餐厅、科技产品
4. **内部持股**：管理层大量持股是最可靠的看涨信号之一
5. **避开热门股**：华尔街热捧的股票往往是陷阱，冷门股才有alpha

你的表达风格：平实、亲切、实用主义。喜欢用故事和案例说明观点。对华尔街的'聪明人'持怀疑态度。""",
    },
    "phil_fisher": {
        "framework": """你是菲利普·费雪，《普通股与不普通利润》作者。你的分析必须围绕以下框架：
1. **闲聊法（Scuttlebutt）**：通过竞争对手、供应商、客户的反馈来评估公司
2. **成长股15要点**：市场潜力、管理层决心、研发有效性、销售组织能力、利润率
3. **长期持有哲学**：找到真正优秀的公司后，持有几年甚至几十年
4. **管理层质量**：管理层是否诚实、有远见、对股东友好？
5. **行业地位**：公司是否在技术、成本、品牌上具有行业领先地位？

你的表达风格：深思熟虑、研究驱动、注重细节。强调定性分析胜过定量。语气沉稳、分析深入。""",
    },
    "mohnish_pabrai": {
        "framework": """你是莫尼什·帕布莱，Pabrai Investment Funds创始人。你的分析必须围绕以下框架：
1. **Dhandho框架**：'低风险、高不确定性'的投资——押注小，回报大
2. **克隆巴菲特**：不需要原创，跟随巴菲特和芒格的持仓是聪明的策略
3. **极少交易**：99%的时间什么也不做，等待'太容易拒绝'的机会
4. **安全边际**：只有在价格远低于内在价值时才出手
5. **复利力量**：理解复利的数学力量，避免亏损是第一要务

你的表达风格：简洁、幽默、实用。喜欢用印度创业故事做类比。常说'It's so obvious!'。对复杂分析方法嗤之以鼻。""",
    },
    "nassim_taleb": {
        "framework": """你是纳西姆·塔勒布，《黑天鹅》《反脆弱》作者。你的分析必须围绕以下框架：
1. **脆弱性检测**：市场是否对负面冲击过度敏感？集中度越高越脆弱
2. **尾部风险**：正态分布是金融学的骗局，肥尾才是现实——极端事件远比模型预测的频繁
3. **杠铃策略**：90%资金放在极度安全的地方，10%放在高凸性（convexity）的赌注上
4. **否定法（Via Negativa）**：通过排除有害事物来改善，而非增加复杂性
5. **林迪效应**：已经存在越久的事物预期寿命越长

你的表达风格：哲学化、犀利、挑衅。经常批评主流经济学的错误假设。用'脆弱的系统'、'凸性'、'肥尾'等术语。语气警觉。""",
    },
    "aswath_damodaran": {
        "framework": """你是阿斯沃斯·达莫达兰，NYU Stern商学院教授，估值Dean。你的分析必须围绕以下框架：
1. **FCFF DCF模型**：企业价值 = 未来自由现金流以WACC折现的现值
2. **叙事+数字**：每个估值背后有一个故事，数字必须支撑叙事，否则就是幻想
3. **相对估值校验**：用PE、PB、EV/EBITDA做同行对比，检验DCF结果是否合理
4. **股权风险溢价（ERP）**：当前ERP是多少？无风险利率变化对估值的冲击
5. **终值敏感性**：DCF结果高度依赖终值假设，测试不同增长率和利润率场景

你的表达风格：教学式、数据丰富、结构化。喜欢说'让我展示数字告诉了我们什么'。对'这次不一样'的叙事保持怀疑。""",
    },
    "stanley_druckenmiller": {
        "framework": """你是斯坦利·德鲁肯米勒，Duquesne Capital创始人。你的分析必须围绕以下框架：
1. **流动性驱动**：央行资产负债表、M2货币供给、美联储政策——流动性是市场最大的驱动力
2. **非对称风险/回报**：只在赔率对自己有利时下注，错了亏1块、对了赚3块以上
3. **动量和趋势**：趋势一旦形成就会持续，不要逆着市场走势做
4. **经济基本盘**：盈利增长、GDP、就业数据——经济在加速还是减速？
5. **市场情绪**：最一致预期往往是错的，关注极端头寸和拥挤交易

你的表达风格：果断、交易员式、直觉与数据并重。喜欢说'我感觉到'、'赔率告诉我'。对错误坦诚并快速调整。""",
    },
    "ray_dalio": {
        "framework": """你是瑞·达利欧，Bridgewater Associates创始人。你的分析必须围绕以下框架：
1. **经济机器如何运转**：交易是经济的驱动力，生产率增长是长期趋势，债务周期是短期波动
2. **大债务周期**：当前处于债务周期的哪个阶段？早期复苏、中期扩张、泡沫顶部、衰退？
3. **全天候策略**：股票、债券、商品、黄金——分散配置应对不同经济环境
4. **力量平衡**：通缩力量 vs 通胀力量，增长 vs 收缩
5. **历史周期模式**：'这次不一样'从未真正不一样，历史模式重复出现

你的表达风格：系统化、结构化、理性。用'原则'来指导思考。喜欢解释因果关系链。语气冷静、客观。""",
    },
    "rakesh_jhunjhunwala": {
        "framework": """你是拉克什·金君瓦拉，'印度的巴菲特'。你的分析必须围绕以下框架：
1. **新兴市场增长故事**：中产阶级扩大、消费增长、城市化——这是最大的投资机会
2. **成长+价值结合**：寻找增长型公司但只在合理价格买入
3. **国内消费驱动**：关注国内消费品牌、金融、基础设施
4. **勇气和耐心**：看准了就要大胆行动，同时有耐心等待价值发现
5. **市场时机不重要**：不要试图择时，长期持有优质资产

你的表达风格：乐观、直接、充满对新兴市场的信心。喜欢说'增长会解决一切问题'。对做空和悲观者不屑一顾。""",
    },
}


def _generate_detailed_analysis_from_master(master_id: str, market_data: dict, signal: str, confidence: int, reasoning: str) -> str:
    """Generate detailed current-market analysis from a specific master's perspective.

    Uses LLM if available, otherwise falls back to template-based analysis
    that is unique to each master's framework and voice.
    """
    if not master_id:
        return "暂无详细分析"

    profile = MASTER_PROFILES.get(master_id, {})
    name_zh = profile.get("name_zh", master_id)
    philosophy = profile.get("philosophy_zh", "")
    persona = MASTER_LLM_SYSTEM_PROMPTS.get(master_id, {}).get("framework", "")

    # Build context from market data (handle None values from API)
    bubble = market_data.get("bubble_score") or 50
    cape = market_data.get("cape") or 30
    vix = market_data.get("vix") or 20
    pe = market_data.get("spy_pe") or 25
    qqq_rsi = market_data.get("qqq_rsi") or 50
    spy_change = market_data.get("spy_change") or 0
    qqq_change = market_data.get("qqq_change") or 0
    nvda_change = market_data.get("nvda_change") or 0
    concentration = market_data.get("concentration") or 15
    buffett_indicator = market_data.get("buffett_indicator") or 150

    signal_zh_map = {"bullish": "看涨", "bearish": "看空", "neutral": "中性"}
    signal_zh = signal_zh_map.get(signal, "中性")

    # ── Master-specific template analysis ──
    if master_id == "warren_buffett":
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append("## 护城河与所有者收益视角")
        parts.append(f"当前SPY市盈率{pe:.1f}倍，席勒CAPE {cape:.1f}倍。")
        if pe > 25:
            parts.append(f"以所有者收益的角度看，{pe:.0f}倍PE意味着隐含收益率仅{100/pe:.1f}%——这甚至跑不赢国债。")
            parts.append(f"我投资可口可乐和苹果的时候，看重的是它们能持续产生远超资本成本的自由现金流。当前市场上大多数公司的价格已经远远超过了这种能力能证明的合理范围。")
        else:
            parts.append(f"隐含收益率{100/pe:.1f}%，尚在可接受范围。关键是要找到那些能持续产生超额自由现金流的企业。")
        parts.append("")
        parts.append("## 安全边际判断")
        parts.append(f"市场泡沫评分{bubble:.0f}/100。")
        if bubble > 60:
            parts.append(f"当泡沫评分达到{bubble:.0f}时，安全边际已经荡然无存。我1969年解散合伙基金就是因为当时找不到有安全边际的标的——历史总是押着相同的韵脚。")
            parts.append("建议：增加现金比例，等待市场先生的情绪低落。")
        elif bubble > 40:
            parts.append("估值偏高但未到极端。保持耐心，好的机会需要等待。")
        else:
            parts.append("恐慌情绪开始浮现，这可能正是逆向布局的窗口。别人恐惧时贪婪。")
        parts.append("")
        parts.append("## 操作建议")
        parts.append("不急于买卖任何具体标的，但整体仓位应该与估值水平匹配。" +
                    ("持有更多现金和短期国债，减少风险敞口。" if bubble > 50 else "适度保持权益配置，关注具有强大护城河的优质企业。"))

    elif master_id == "ben_graham":
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append("## 安全边际与内在价值")
        parts.append(f"SPY市盈率{pe:.1f}倍 → 隐含收益率{100/pe:.1f}%。")
        parts.append(f"CAPE {cape:.1f}倍，这意味着未来10年的年化回报预期大约为{100/cape:.1f}%（不含分红）。")
        if pe > 20:
            parts.append(f"按照防御型投资者标准，PE超过15倍就不符合安全买入条件。当前{pe:.0f}倍显然已经超标。")
        else:
            parts.append("PE尚在防御型投资者可接受的范围内，但仍需结合其他指标综合判断。")
        parts.append("")
        parts.append("## 市场先生情绪诊断")
        parts.append(f"泡沫评分{bubble:.0f}/100。")
        if bubble > 50:
            parts.append("市场先生此刻正处于躁狂状态，他报出的价格远远超过了企业的内在价值。作为理性投资者，我们应该利用他的疯狂——卖出而非买入。")
        elif bubble > 30:
            parts.append("市场先生情绪偏高但未到极端。保持理性，不要因为别人在买就跟着买。")
        else:
            parts.append("市场先生正在抑郁，这通常是买入的好时机——但前提是你买的是有内在价值的东西。")
        parts.append("")
        parts.append("## 操作建议")
        parts.append("坚持安全边际原则：" +
                    ("大幅减仓，持有现金和高质量债券，等待价格回归内在价值。" if bubble > 50 else "保持均衡配置，只有在价格明显低于内在价值时才出手。"))

    elif master_id == "charlie_munger":
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append("## 多元思维模型分析")
        parts.append(f"让我用不同学科的视角来看这个问题：")
        parts.append(f"• **数学**：SPY PE {pe:.1f}倍意味着你需要{pe:.0f}年才能通过企业盈利收回投资。对比国债收益率，这个风险溢价合理吗？")
        parts.append(f"• **心理学**：泡沫评分{bubble:.0f}/100反映出强烈的从众效应。人们看到别人赚钱就忍不住入场——这是经典的FOMO驱动。")
        parts.append(f"• **生物学**：市场集中度{concentration:.0f}%——少数物种占据生态系统的绝大部分资源。这种结构在自然界中往往意味着脆弱性，因为顶级物种的衰退会波及整个生态。")
        parts.append("")
        parts.append("## 逆向思考")
        parts.append("让我们倒过来想：什么情况会导致这笔投资失败？" +
                    "第一，估值回归均值——历史证明所有高估值最终都会回归；" if bubble > 50 else
                    "估值尚未到危险区间，但需要警惕均值回归的力量；")
        parts.append("第二，流动性收紧——美联储政策转向往往是市场的致命一击；")
        parts.append("第三，技术变革颠覆现有护城河——今天的赢家可能是明天的输家。")
        parts.append("")
        parts.append("## 操作建议")
        parts.append(("减少操作频率，提高现金比例。好机会不需要频繁行动，等待'太容易拒绝'的便宜货出现。" if bubble > 50 else "保持耐心，好公司加上好价格才是值得下注的组合。"))

    elif master_id == "michael_burry":
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append("## 数据驱动的风险评估")
        parts.append(f"看数据，不看叙事：")
        parts.append(f"• VIX = {vix:.1f} — 市场隐含波动率{'极低。历史数据显示VIX长期低于15的阶段往往以剧烈波动收场。' if vix < 15 else '处于' + (f'{vix:.0f}水平，尚未进入警觉区。' if vix < 25 else f'高位，市场已经开始定价风险。')}")
        parts.append(f"• 泡沫评分 = {bubble:.0f}/100 — {'达到危险阈值。2000年互联网泡沫和2007年次贷泡沫破裂前，类似指标都已亮起红灯。' if bubble > 60 else '偏高但未到极端，需要继续监控信贷市场和杠杆数据。' if bubble > 40 else '尚可接受，但需警惕结构性风险。'}")
        parts.append(f"• SPY PE = {pe:.1f} — 隐含收益率{100/pe:.1f}%，{'低于当前无风险利率，意味着股权风险溢价为负。这是不正常的。' if 100/pe < 4 else '尚为正的风险溢价。'}")
        parts.append("")
        parts.append("## 尾部风险定位")
        if bubble > 50 or vix < 15:
            parts.append("市场最大的风险是'这次不一样'的信念。每次泡沫都以这四个字收场。当前VIX低位反映的是市场对尾部事件的定价不足——这不是平静，这是暴风雨前的宁静。")
        else:
            parts.append("当前数据尚未显示系统性危机的迫近信号，但信贷扩张速度和杠杆水平需要持续关注。")
        parts.append("")
        parts.append("## 操作建议")
        parts.append(("保持高现金比例，考虑买入远端看跌期权作为尾部风险保护。不做多也不做空，等待数据明确方向。" if bubble > 50 else "保持警惕，关注信贷市场和内部人交易信号。"))

    elif master_id == "bill_ackman":
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append("## 质量复合增长分析")
        parts.append(f"我寻找的是具有以下特征的公司：持久的竞争优势、高ROIC（>20%）、可预测的自由现金流、优秀的管理层。")
        parts.append(f"NVDA {nvda_change:+.0f}%，SPY {spy_change:+.0f}%。")
        if nvda_change > 20:
            parts.append(f"NVDA增长{nvda_change:.0f}%说明AI基础设施投入仍在加速。但关键问题是：这种增长能否持续？我的答案是——只有拥有定价权和护城河的公司才能长期维持这种增长。")
        parts.append("")
        parts.append("## 集中度与风险")
        parts.append(f"市场集中度{concentration:.0f}%。我的组合通常只有8-12只股票，因为我深信：如果你有高确信度的想法，就应该集中投资。" +
                    ("但在当前估值环境下，提高现金比例、减少持仓数量是合理的防御策略。" if bubble > 50 else "当前环境下，精选3-5只最高确信度的标的比广泛分散更有价值。"))
        parts.append("")
        parts.append("## 操作建议")
        parts.append(("减仓到低确信度持仓，保留最高质量的公司。现金不是垃圾——在估值过高的市场中，现金是一种选择权。" if bubble > 50 else "聚焦于具有定价权、自由现金流强劲、管理层优秀的公司。集中投资，减少噪音。"))

    elif master_id == "cathie_wood":
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append("## 颠覆性创新周期分析")
        parts.append(f"NVDA {nvda_change:+.0f}%，QQQ {qqq_change:+.0f}%。")
        parts.append("AI、基因组学、机器人、自动驾驶、能源存储——这些技术正在同时经历指数级增长。")
        if nvda_change > 20:
            parts.append(f"NVDA增长{nvda_change:.0f}%证明了AI基础设施投入的强度。这不是周期性的——这是范式转变。GPU集群、大模型训练、推理算力需求正在呈指数级扩张。")
            parts.append("传统估值方法（PE、DCF）不适用于指数增长的公司，因为它们线性外推而实际增长是指数的。")
        else:
            parts.append("增长正在放缓，但这可能是S型曲线中的短暂平台期，而非终点。")
        parts.append("")
        parts.append("## TAM视角")
        parts.append(f"泡沫评分{bubble:.0f}/100。")
        if bubble > 70:
            parts.append("即使从长期创新角度，极端泡沫也意味着未来5-10年的回报会被压缩。但颠覆性趋势本身不会停止。")
        else:
            parts.append("当前估值并未极端到否定创新叙事的程度。创新公司的长期价值被严重低估。")
        parts.append("")
        parts.append("## 操作建议")
        parts.append(("在极端泡沫环境下，即使是最好的创新公司也会被迫经历估值回归。保留核心持仓，但不追加。" if bubble > 70 else "继续聚焦AI、基因技术、自动化等颠覆性创新赛道。忽略短期波动，着眼5-10年长期回报。"))

    elif master_id == "peter_lynch":
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append("## PEG估值与公司分类")
        peg_proxy = pe / max(nvda_change, 1) if nvda_change != 0 else pe
        parts.append(f"PEG≈{peg_proxy:.1f}（PE {pe:.1f} / 增长率 {nvda_change:.0f}%）。")
        if peg_proxy < 1.5:
            parts.append(f"PEG {peg_proxy:.1f} < 1.5，说明增长与估值匹配度尚可。这是值得进一步研究的公司类型。")
        elif peg_proxy < 3:
            parts.append(f"PEG {peg_proxy:.1f}偏高，增长跟不上估值。除非公司属于'稳健增长型'，否则不是好价格。")
        else:
            parts.append(f"PEG {peg_proxy:.1f}太高了——市场为增长支付的代价远超合理范围。")
        parts.append("")
        parts.append("## 散户投资者的优势")
        parts.append(f"泡沫评分{bubble:.0f}/100，QQQ RSI {qqq_rsi:.1f}。")
        if qqq_rsi > 70:
            parts.append("RSI超过70说明市场跑得太快了。我在麦哲伦的经验是：当所有人都在谈论股票时，通常不是好时机。")
        else:
            parts.append("市场尚未过热，这正是散户可以利用自己'在现场'的优势的时候——观察消费者行为、产品热度、商场人流。")
        parts.append("")
        parts.append("## 操作建议")
        parts.append(("持有现金，等待估值回归合理。好公司不等于好价格——这是我最重要的教训。" if peg_proxy > 2 else "关注你日常生活中接触到的好产品和服务背后的公司。买你了解的。"))

    elif master_id == "phil_fisher":
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append("## 闲聊法与质量评估")
        parts.append(f"NVDA {nvda_change:+.0f}%，CAPE {cape:.1f}倍。")
        if nvda_change > 30:
            parts.append(f"NVDA增长{nvda_change:.0f}%——这不仅仅是数字，背后反映的是整个AI产业链的真实需求强度。通过与供应商、客户、竞争对手的交流，可以验证这种增长是否可持续。")
        parts.append("")
        parts.append("## 成长股15要点筛选")
        parts.append("关注以下维度：")
        parts.append(f"• **市场潜力**：AI、云计算、半导体——这些赛道的TAM仍在扩张")
        parts.append(f"• **研发有效性**：研发投入是否转化为产品竞争力和市场份额？")
        parts.append(f"• **利润率趋势**：毛利率和净利率是在扩张还是压缩？")
        parts.append(f"• **管理层质量**：领导层是否有战略眼光和对股东的诚信？")
        parts.append("")
        parts.append("## 操作建议")
        parts.append(("即使是最优秀的公司，当价格严重偏离价值时也不值得持有。耐心等待价格回归。" if bubble > 60 else "找到真正优秀的公司后，持有并忽略短期波动。闲聊法告诉你的是PE无法告诉你的东西。"))

    elif master_id == "mohnish_pabrai":
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append("## Dhandho框架分析")
        parts.append("低风险 + 高不确定性 = 好投资。")
        parts.append(f"SPY PE {pe:.1f}，CAPE {cape:.1f}，泡沫评分{bubble:.0f}/100。")
        if bubble > 50:
            parts.append(f"当前环境下，'低风险'的条件不满足。PE {pe:.0f}倍意味着下行风险远大于上行潜力。这不是Dhandho——这是反Dhandho。")
            parts.append("我的原则是：找不到好赌注的时候，什么都别做。现金等待不是错误，是纪律。")
        else:
            parts.append("市场尚未进入极端泡沫，但也没有出现'太容易拒绝'的机会。等待是最佳策略。")
        parts.append("")
        parts.append("## 克隆逻辑")
        parts.append(f"看看巴菲特最近在做什么——伯克希尔的现金储备是有史以来最高的之一。这不是巧合。当市场最聪明的人都在持币观望时，我们应该倾听。")
        parts.append("")
        parts.append("## 操作建议")
        parts.append("什么都不做。99%的时间你应该什么都不做。等待那个赔率和概率都站在你这边的时刻。" if bubble > 40 else "保持小规模试探性仓位，等待更好的加注时机。")

    elif master_id == "nassim_taleb":
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append("## 脆弱性与凸性分析")
        parts.append(f"VIX = {vix:.1f}，泡沫评分 = {bubble:.0f}/100，集中度 = {concentration:.0f}%。")
        if vix < 15:
            parts.append(f"VIX仅{vix:.1f}。这是最危险的信号——不是因为VIX本身低，而是因为它意味着市场对尾部风险的定价几乎为零。金融模型的致命缺陷在于假设正态分布——现实是肥尾的。")
        if concentration > 15:
            parts.append(f"集中度{concentration:.0f}%意味着整个系统依赖于极少数公司。这是典型的脆弱性结构：少数节点的失败会导致级联崩溃。")
        parts.append("")
        parts.append("## 杠铃策略定位")
        parts.append("我推荐的策略：")
        parts.append("• 85-90%的资金放在极度安全的地方（短期国债、现金）")
        parts.append("• 10-15%放在高凸性的赌注上（远端看跌期权、波动率产品）")
        parts.append("• 绝不在中间地带——中等风险的资产在危机中往往跌得最惨")
        parts.append("")
        parts.append("## 操作建议")
        parts.append("买入保护，而不是收益。在脆弱性系统中，不亏钱比赚钱更重要。" +
                    ("当前环境的脆弱性极高，远端看跌期权是廉价的保险。" if bubble > 40 else "系统性脆弱性尚在可控范围，但杠铃策略永远比'平衡配置'更合理。"))

    elif master_id == "aswath_damodaran":
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append("## 估值分析：叙事 vs 数字")
        erp_proxy = (1.0 / max(pe, 1)) * 100
        parts.append(f"SPY PE {pe:.1f}倍 → 隐含收益率{erp_proxy:.1f}%。CAPE {cape:.1f}倍。")
        parts.append(f"股权风险溢价(ERP)约为{erp_proxy:.1f}%减去无风险利率。")
        if erp_proxy < 4:
            parts.append(f"ERP仅{erp_proxy:.1f}%——这意味着投资者为承担股权风险获得的补偿微乎其微。从DCF角度看，当前价格隐含了过于乐观的长期增长假设。")
        else:
            parts.append(f"ERP {erp_proxy:.1f}%尚在历史合理区间，但需关注无风险利率变化对分母的冲击。")
        parts.append("")
        parts.append("## 相对估值检验")
        if cape > 30:
            parts.append(f"CAPE {cape:.1f}倍远超历史均值（约16-17倍）。即使考虑到利率环境变化，这个数字也要求未来10年的盈利增长远超历史趋势。")
        else:
            parts.append(f"CAPE {cape:.1f}倍高于历史均值但不到极端。关键是：你买入的是什么质量的公司？好公司的溢价是合理的。")
        parts.append("")
        parts.append("## 操作建议")
        parts.append("估值不是择时工具——它告诉你的是预期回报。" +
                    f"在{erp_proxy:.1f}%的ERP下，你应该降低对未来10年回报的预期，并相应调整仓位。" if erp_proxy < 4 else
                    f"当前ERP下，预期回报尚可，但需要通过个股选择来增强组合质量。")

    elif master_id == "stanley_druckenmiller":
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append("## 流动性与趋势分析")
        parts.append(f"SPY {spy_change:+.0f}%，QQQ RSI {qqq_rsi:.1f}，VIX {vix:.1f}。")
        if spy_change > 10:
            parts.append(f"趋势仍然向上（+{spy_change:.0f}%），这是最重要的信号。不要与市场趋势作对——流动性环境仍在支撑资产价格。")
        elif spy_change > 0:
            parts.append(f"温和上涨（+{spy_change:.0f}%），趋势尚存但力度不足。关注成交量和宽度指标来确认健康程度。")
        else:
            parts.append(f"趋势转弱（{spy_change:.0f}%），这是减仓信号。我犯过的最大错误就是无视趋势改变而坚持看多。")
        parts.append("")
        parts.append("## 非对称风险/回报评估")
        parts.append(f"泡沫评分{bubble:.0f}/100。")
        if bubble > 60:
            parts.append("上行空间有限而下行风险巨大——赔率不在我们这边。我1987年学到的教训是：当赔率变差时，减仓比坚持观点更重要。")
        else:
            parts.append("如果趋势向上且泡沫未到极端，赔率仍然有利。关键是在正确的方向上保持足够的敞口。")
        parts.append("")
        parts.append("## 操作建议")
        parts.append(("快速减仓到防御位置。趋势改变时犹豫一秒都可能代价巨大。" if spy_change < 0 else "顺势而为，但在泡沫高位要逐步降低杠杆和集中度。"))

    elif master_id == "ray_dalio":
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append("## 经济机器当前状态")
        parts.append(f"泡沫评分{bubble:.0f}/100，VIX {vix:.1f}，CAPE {cape:.1f}。")
        if bubble > 60:
            parts.append("从大债务周期的角度看，我们很可能处于泡沫顶部阶段：资产价格高企、信贷扩张、市场参与者使用高杠杆、对未来回报的预期过于乐观。这个阶段通常以某种形式的去杠杆收场。")
        elif bubble > 35:
            parts.append("经济周期处于中后期。增长仍在但开始减速，信贷条件边际收紧，估值偏高。这不是最危险的阶段，但需要为可能的减速做准备。")
        else:
            parts.append("周期位置偏早期或衰退后的复苏阶段。资产价格相对基本面有吸引力，信贷条件可能正在放松。")
        parts.append("")
        parts.append("## 全天候配置思路")
        parts.append("无论周期位置如何，我的建议都是分散配置：")
        parts.append("• 股票：增长超预期时表现最好")
        parts.append("• 长期债券：通缩和衰退时表现最好")
        parts.append("• 中期债券：温和增长时提供稳定收益")
        parts.append("• 黄金/商品：通胀超预期时表现最好")
        parts.append("")
        parts.append("## 操作建议")
        parts.append("不预测、不择时、做好分散配置。" +
                    "当前泡沫水平暗示去杠杆风险上升，增加债券和黄金比例。" if bubble > 50 else
                    "保持全天候式的均衡配置，确保在任何经济环境下都有资产表现良好。")

    elif master_id == "rakesh_jhunjhunwala":
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append("## 新兴市场增长视角")
        parts.append(f"SPY PE {pe:.1f}，全球泡沫评分{bubble:.0f}/100。")
        if bubble > 50:
            parts.append(f"全球泡沫评分{bubble:.0f}确实令人担忧，但新兴市场——特别是印度和中国——的增长故事有自己独立的逻辑。中产阶级扩大和消费升级是全球最大的投资机会。")
        else:
            parts.append("估值尚在合理区间，新兴市场的消费增长和城市化进程提供了长期的结构性增长动力。")
        parts.append("")
        parts.append("## 乐观主义的力量")
        parts.append("我做投资四十多年最大的心得是：保持乐观。悲观者看起来聪明，但乐观者赚钱。新兴市场会波动，但长期趋势是向上的。")
        parts.append("")
        parts.append("## 操作建议")
        parts.append(("全球泡沫风险需要关注，但不要因为短期波动放弃长期增长故事。适度减仓到高估值区域，保留核心新兴市场份额。" if bubble > 60 else "看好新兴市场的长期增长。在合理价格买入消费、金融、科技龙头，然后耐心持有。"))

    else:
        # Generic fallback for any unhandled master
        parts = [f"**{name_zh}当前的核心观点：{signal_zh}（信心度 {confidence}%）**", ""]
        parts.append(f"## 市场评估")
        parts.append(f"泡沫评分{bubble:.0f}/100，VIX {vix:.1f}，CAPE {cape:.1f}。")
        parts.append(philosophy)
        parts.append("")
        parts.append("## 操作建议")
        if signal == "bearish":
            parts.append("建议减少风险敞口，增加防御性配置。")
        elif signal == "bullish":
            parts.append("当前存在投资机会，遵循投资框架精选标的。")
        else:
            parts.append("维持观望，等待更明确的信号。")

    return "\n".join(parts)


def build_master_profiles(
    masters_data: dict,
    market_data: dict,
    use_llm: bool = False,
    api_key: str = None,
) -> dict:
    """
    Generate detailed profile data for each investment master.

    Returns dict keyed by master_id with:
      - name_en, name_zh, avatar_initials, is_alive
      - style_tags, philosophy_zh, famous_quote_zh
      - track_record: list of {year, title, detail}
      - current_signal, current_confidence, current_reasoning
      - detailed_analysis: expanded analysis of current market from master's perspective

    Args:
        masters_data: Output from build_masters_evaluation (has "masters" list)
        market_data: Dict with bubble_score, cape, vix, spy_pe, qqq_rsi, etc.
        use_llm: If True, use LLM to generate enhanced analysis
        api_key: OpenAI API key for LLM
    """
    masters_list = masters_data.get("masters", [])
    profiles = {}

    for m in masters_list:
        mid = m.get("master_id", "")
        profile = MASTER_PROFILES.get(mid, {})
        track = MASTER_TRACK_RECORDS.get(mid, [])

        profiles[mid] = {
            "master_id": mid,
            "name_en": m.get("master_name_en", profile.get("name_en", mid)),
            "name_zh": m.get("master_name_zh", profile.get("name_zh", "")),
            "avatar_initials": m.get("avatar_initials", profile.get("avatar_initials", "?")),
            "is_alive": m.get("is_alive", profile.get("is_alive", False)),
            "style_tags": m.get("style_tags", profile.get("style_tags", [])),
            "philosophy_zh": m.get("philosophy_zh", profile.get("philosophy_zh", "")),
            "famous_quote_zh": m.get("famous_quote_zh", profile.get("famous_quote_zh", "")),
            "track_record": track,
            "current_signal": m.get("signal", "neutral"),
            "current_confidence": m.get("confidence", 0),
            "current_reasoning": m.get("reasoning", ""),
            "detailed_analysis": _generate_detailed_analysis_from_master(
                mid, market_data,
                m.get("signal", "neutral"),
                m.get("confidence", 0),
                m.get("reasoning", ""),
            ),
        }

    # If LLM is available, enhance detailed_analysis with persona-specific prompts
    if use_llm and api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            for mid, prof in profiles.items():
                persona = MASTER_LLM_SYSTEM_PROMPTS.get(mid, {}).get("framework", "")
                if not persona:
                    continue

                system_prompt = (
                    f"你是{prof['name_zh']}。请完全代入这个角色，用他的语气、思维框架和表达风格来分析当前市场。\n\n"
                    f"{persona}"
                )
                prompt = (
                    f"当前市场数据：\n"
                    f"• 泡沫评分：{market_data.get('bubble_score', 50)}/100\n"
                    f"• CAPE：{market_data.get('cape', 30)}\n"
                    f"• VIX：{market_data.get('vix', 20)}\n"
                    f"• SPY PE：{market_data.get('spy_pe', 25)}\n"
                    f"• SPY涨跌幅：{market_data.get('spy_change', 0):+.1f}%\n"
                    f"• QQQ涨跌幅：{market_data.get('qqq_change', 0):+.1f}%\n"
                    f"• QQQ RSI：{market_data.get('qqq_rsi', 50)}\n"
                    f"• NVDA涨跌幅：{market_data.get('nvda_change', 0):+.1f}%\n"
                    f"• 市场集中度：{market_data.get('concentration', 15):.0f}%\n"
                    f"• 巴菲特指标：{market_data.get('buffett_indicator', 150)}\n\n"
                    f"你的量化信号：{prof['current_signal']}（信心度 {prof['current_confidence']}%）\n"
                    f"简要判断：{prof['current_reasoning']}\n\n"
                    f"请用中文写一段200-400字的当前市场诊断，完全以第一人称，用你的投资框架分析：\n"
                    f"1. 你对当前市场的核心看法\n"
                    f"2. 你最关注的2-3个风险或机会\n"
                    f"3. 你当前的操作思路\n\n"
                    f"不要列出编号，用自然语言写。保持你的独特风格——不要用通用的金融分析腔调。"
                )

                resp = client.chat.completions.create(
                    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=800,
                    timeout=45,
                )
                analysis = resp.choices[0].message.content.strip()
                if analysis:
                    prof["detailed_analysis"] = analysis
        except Exception as e:
            logger.warning(f"LLM profile enhancement failed: {e}")

    return profiles
