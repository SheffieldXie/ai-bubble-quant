"""
Signal Monitor Module.
Tracks predefined trigger signals for entering 0DTE option trades.
Focus: A-share / China Financial Futures Exchange (CFFEX) instruments.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SignalSeverity(Enum):
    INFO = "INFO 正常"
    WATCH = "WATCH 关注"
    ALERT = "ALERT 警告"
    TRIGGER = "TRIGGER 触发"


class SignalCategory(Enum):
    MACRO = "Macro"
    TECHNICAL = "Technical"
    SENTIMENT = "Sentiment"
    FUNDAMENTAL = "Fundamental"
    EVENT = "Event"


@dataclass
class TriggerSignal:
    """A single monitorable signal."""
    name: str
    category: SignalCategory
    description: str
    current_value: Optional[float] = None
    threshold: float = 0.0  # Value that triggers the signal
    direction: str = "above"  # "above" or "below"
    severity: SignalSeverity = SignalSeverity.INFO
    weight: float = 1.0  # How important this signal is (0-1)
    triggered: bool = False
    last_checked: Optional[str] = None

    def check(self, value: float) -> bool:
        """Check if the signal is triggered."""
        self.current_value = value
        self.last_checked = datetime.now().isoformat()

        if self.direction == "above":
            self.triggered = value >= self.threshold
        else:
            self.triggered = value <= self.threshold

        if self.triggered:
            self.severity = SignalSeverity.TRIGGER
        elif value is not None:
            # How close are we to triggering?
            if self.direction == "above":
                proximity = value / self.threshold
            else:
                proximity = self.threshold / max(value, 0.001)

            if proximity >= 0.9:
                self.severity = SignalSeverity.ALERT
            elif proximity >= 0.75:
                self.severity = SignalSeverity.WATCH
            else:
                self.severity = SignalSeverity.INFO

        return self.triggered


@dataclass
class SignalDashboard:
    """Collection of all monitored signals."""
    signals: list[TriggerSignal] = field(default_factory=list)

    def add(self, signal: TriggerSignal):
        self.signals.append(signal)

    def update(self, signal_name: str, value: float) -> bool:
        """Update a signal's value and check if triggered."""
        for s in self.signals:
            if s.name == signal_name:
                return s.check(value)
        return False

    def triggered_signals(self) -> list[TriggerSignal]:
        return [s for s in self.signals if s.triggered]

    def alert_level(self) -> SignalSeverity:
        """Get the highest severity level among all signals."""
        max_sev = SignalSeverity.INFO
        for s in self.signals:
            if self._sev_rank(s.severity) > self._sev_rank(max_sev):
                max_sev = s.severity
        return max_sev

    @staticmethod
    def _sev_rank(sev: SignalSeverity) -> int:
        return {
            SignalSeverity.INFO: 0,
            SignalSeverity.WATCH: 1,
            SignalSeverity.ALERT: 2,
            SignalSeverity.TRIGGER: 3,
        }.get(sev, 0)

    def summary_score(self) -> float:
        """
        Calculate overall readiness score (0-100).
        Higher = more signals triggered = better time to enter trade.
        Only counts triggered signals; proximity bonus applies only when
        at least one signal has already fired.
        """
        if not self.signals:
            return 0

        total_weight = sum(s.weight for s in self.signals)
        triggered_weight = sum(s.weight for s in self.signals if s.triggered)

        # No triggered signals → score is 0 (no false positives from proximity)
        if triggered_weight == 0:
            return 0.0

        # Proximity bonus: how close non-triggered signals are to threshold
        # Only applies as a small bonus on top of already-triggered signals
        proximity_score = 0
        for s in self.signals:
            if not s.triggered and s.current_value is not None:
                if s.direction == "above":
                    p = min(1.0, s.current_value / s.threshold) if s.threshold > 0 else 0
                else:
                    p = min(1.0, s.threshold / max(s.current_value, 0.001)) if s.current_value > 0 else 0
                proximity_score += p * s.weight

        # 85% triggered + 15% proximity bonus (only when triggered > 0)
        score = (0.85 * triggered_weight + 0.15 * proximity_score) / total_weight * 100
        return round(score, 1)

    def format_report(self) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("CFFEX 0DTE OPTION ENTRY SIGNAL DASHBOARD")
        lines.append("=" * 80)
        lines.append(f"\nReadiness Score: {self.summary_score():.1f}/100")
        lines.append(f"Alert Level: {self.alert_level().value}")
        lines.append(f"Signals Triggered: {len(self.triggered_signals())}/{len(self.signals)}")
        lines.append("")
        lines.append(f"{'Signal':<35} {'Category':<14} {'Value':>10} {'Threshold':>10} {'Status':<12}")
        lines.append("-" * 80)

        for s in sorted(self.signals, key=lambda x: self._sev_rank(x.severity), reverse=True):
            val = f"{s.current_value:.2f}" if s.current_value is not None else "N/A"
            thresh = f"{s.threshold:.2f}"
            status = "✅ TRIGGERED" if s.triggered else s.severity.value
            dir_symbol = "↑" if s.direction == "above" else "↓"
            lines.append(f"{s.name:<35} {s.category.value:<14} {val:>10} {thresh:>10} {status}")

        triggered = self.triggered_signals()
        if triggered:
            lines.append("")
            lines.append("⚡ TRIGGERED SIGNALS:")
            for s in triggered:
                lines.append(f"   • {s.name}: {s.description}")
        else:
            lines.append("")
            lines.append("No signals triggered yet. Monitoring continues...")

        lines.append("=" * 80)
        return "\n".join(lines)


def build_default_dashboard() -> SignalDashboard:
    """
    Build the default signal dashboard for CFFEX 0DTE option entry.
    Based on historical crash patterns and AI bubble characteristics.
    """
    dashboard = SignalDashboard()

    # ── Macro Signals ──────────────────────────────────────────────────

    dashboard.add(TriggerSignal(
        name="VIX 恐慌指数突破30",
        category=SignalCategory.MACRO,
        description="CBOE VIX恐慌指数突破30，代表市场极度恐慌。通常在大跌或崩盘初期出现，是最直接的入场信号之一。",
        threshold=30.0,
        direction="above",
        weight=1.0,
    ))

    dashboard.add(TriggerSignal(
        name="美债10年期收益率突破4.8%",
        category=SignalCategory.MACRO,
        description="美国10年期国债收益率突破4.8%，代表金融条件收紧，资金成本上升，对高估值科技股构成压力。",
        threshold=4.8,
        direction="above",
        weight=0.8,
    ))

    dashboard.add(TriggerSignal(
        name="美元兑人民币突破7.35",
        category=SignalCategory.MACRO,
        description="人民币汇率突破7.35，代表资本外流压力加大，可能引发A股外资撤离，加速下跌。",
        threshold=7.35,
        direction="above",
        weight=0.7,
    ))

    # ── Technical Signals ─────────────────────────────────────────────

    dashboard.add(TriggerSignal(
        name="纳斯达克RSI周线超买>80",
        category=SignalCategory.TECHNICAL,
        description="纳斯达克100指数RSI在周线级别突破80，代表极度超买，历史上这是回调的前兆。",
        threshold=80.0,
        direction="above",
        weight=0.9,
    ))

    dashboard.add(TriggerSignal(
        name="中证1000跌破20日均线",
        category=SignalCategory.TECHNICAL,
        description="中证1000指数跌破20日移动平均线，代表短期趋势反转，是中证1000期权（MO合约）的参考信号。",
        threshold=0.0,
        direction="below",
        weight=0.8,
    ))

    dashboard.add(TriggerSignal(
        name="纳指单日暴跌>3%",
        category=SignalCategory.TECHNICAL,
        description="纳斯达克指数单日下跌超过3%，历史上这往往不是孤立事件，而是更大跌幅的开端。",
        threshold=-3.0,
        direction="below",
        weight=1.0,
    ))

    # ── Sentiment Signals ─────────────────────────────────────────────

    dashboard.add(TriggerSignal(
        name="A股AI概念涨停家数>20",
        category=SignalCategory.SENTIMENT,
        description="单日超过20只AI相关股票涨停，代表市场情绪极度狂热，往往是阶段性见顶的信号。",
        threshold=20.0,
        direction="above",
        weight=0.7,
    ))

    dashboard.add(TriggerSignal(
        name="期权Put/Call比率<0.6",
        category=SignalCategory.SENTIMENT,
        description="看跌/看涨期权比率低于0.6，说明市场几乎无人做空，过度乐观是危险信号。",
        threshold=0.6,
        direction="below",
        weight=0.8,
    ))

    # ── Fundamental Signals ───────────────────────────────────────────

    dashboard.add(TriggerSignal(
        name="英伟达营收增速降至<40%",
        category=SignalCategory.FUNDAMENTAL,
        description="英伟达数据中心业务同比增速低于40%，代表AI基础设施需求放缓，是整个AI叙事的核心基本面指标。",
        threshold=40.0,
        direction="below",
        weight=1.0,
    ))

    dashboard.add(TriggerSignal(
        name="科技巨头削减AI资本开支",
        category=SignalCategory.FUNDAMENTAL,
        description="微软/谷歌/亚马逊等科技巨头下调AI资本开支计划，代表行业投资降温。",
        threshold=0.0,
        direction="below",
        weight=0.9,
    ))

    # ── Event Signals ─────────────────────────────────────────────────

    dashboard.add(TriggerSignal(
        name="美联储释放鹰派信号",
        category=SignalCategory.EVENT,
        description="美联储暗示不降息或加息，流动性收紧，对高估值成长股打击最大。",
        threshold=0.0,
        direction="below",
        weight=0.8,
    ))

    dashboard.add(TriggerSignal(
        name="中国科技监管政策收紧",
        category=SignalCategory.EVENT,
        description="中国出台针对科技/AI公司的监管政策，可能引发A股科技板块恐慌性下跌。",
        threshold=0.0,
        direction="below",
        weight=0.7,
    ))

    return dashboard
