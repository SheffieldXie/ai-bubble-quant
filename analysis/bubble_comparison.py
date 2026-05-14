"""
Bubble Comparison Module.
Compares current AI market conditions to historical bubbles (2000 dotcom, 2008 housing).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BubbleIndicator:
    name: str
    current_value: Optional[float] = None
    dotcom_2000_value: Optional[float] = None
    description: str = ""
    warning_threshold: Optional[float] = None  # Value at which bubble risk is HIGH
    extreme_threshold: Optional[float] = None   # Value at which bubble risk is EXTREME
    unit: str = ""
    interpretation_higher: bool = True  # True = higher is more bubble-like


@dataclass
class BubbleAssessment:
    indicators: list[BubbleIndicator] = field(default_factory=list)

    def add(self, indicator: BubbleIndicator):
        self.indicators.append(indicator)

    def score(self) -> dict:
        """
        Score current bubble risk on a 0-100 scale.
        0 = no bubble, 100 = extreme bubble (like March 2000)
        """
        scores = []
        has_data = False
        for ind in self.indicators:
            if ind.current_value is None:
                scores.append({
                    "name": ind.name,
                    "current": None,
                    "dotcom_2000": ind.dotcom_2000_value,
                    "score": 0,
                    "unit": ind.unit,
                    "description": ind.description,
                    "extreme_threshold": ind.extreme_threshold,
                })
                continue

            has_data = True
            # How far are we from normal to extreme?
            if ind.extreme_threshold and ind.warning_threshold:
                if ind.interpretation_higher:
                    if ind.current_value <= ind.warning_threshold:
                        s = 0
                    elif ind.current_value <= ind.extreme_threshold:
                        s = (ind.current_value - ind.warning_threshold) / \
                            (ind.extreme_threshold - ind.warning_threshold) * 50
                    else:
                        s = 50 + min(50, (ind.current_value - ind.extreme_threshold) / \
                                     ind.extreme_threshold * 50)
                else:
                    # Lower is more bubble-like (e.g., VIX)
                    if ind.current_value >= ind.warning_threshold:
                        s = 0
                    elif ind.current_value >= ind.extreme_threshold:
                        s = (ind.warning_threshold - ind.current_value) / \
                            (ind.warning_threshold - ind.extreme_threshold) * 50
                    else:
                        s = 50 + min(50, (ind.extreme_threshold - ind.current_value) / \
                                     ind.extreme_threshold * 50)
            else:
                s = 50  # Unknown, neutral

            s = max(0, min(100, s))
            scores.append({
                "name": ind.name,
                "current": ind.current_value,
                "dotcom_2000": ind.dotcom_2000_value,
                "score": round(s, 1),
                "unit": ind.unit,
                "description": ind.description,
                "extreme_threshold": ind.extreme_threshold,
            })

        # Average only indicators that have data
        scored_indicators = [s for s in scores if s["current"] is not None]

        if not scored_indicators:
            overall = 50
            verdict = "暂无数据"
        else:
            overall = sum(s["score"] for s in scored_indicators) / len(scored_indicators)

            if overall >= 80:
                verdict = "极高泡沫 —— 与2000年3月峰值相似"
            elif overall >= 65:
                verdict = "高度泡沫 —— 后期泡沫阶段，崩盘风险高"
            elif overall >= 45:
                verdict = "中度泡沫 —— 泡沫条件已出现，密切监控"
            elif overall >= 25:
                verdict = "低度泡沫 —— 存在部分高估，尚未泡沫化"
            else:
                verdict = "正常状态 —— 未检测到泡沫信号"

        return {
            "overall": round(overall, 1),
            "indicators": scores,
            "verdict": verdict,
        }

    def format_report(self) -> str:
        result = self.score()

        lines = []
        lines.append("=" * 80)
        lines.append("AI BUBBLE vs 2000 DOTCOM COMPARISON")
        lines.append("=" * 80)
        lines.append(f"\nOverall Bubble Score: {result['overall']:.1f}/100")
        lines.append(f"Verdict: {result['verdict']}")
        lines.append("")
        lines.append(f"{'Indicator':<30} {'Current':>12} {'Dotcom 2000':>12} {'Score':>8} {'Unit':<10}")
        lines.append("-" * 80)

        for ind in result["indicators"]:
            current = f"{ind['current']:.1f}" if ind["current"] is not None else "N/A"
            dotcom = f"{ind['dotcom_2000']:.1f}" if ind["dotcom_2000"] is not None else "N/A"
            score_bar = "█" * int(ind["score"] / 5) + "░" * (20 - int(ind["score"] / 5))
            lines.append(f"{ind['name']:<30} {current:>12} {dotcom:>12} {ind['score']:>6.1f}  {ind['unit']:<10} {score_bar}")

        lines.append("")
        lines.append("Score bar: ░░░░░░░░░░░░░░░░░░░░ (0) to ████████████████████ (100)")
        lines.append("=" * 80)

        return "\n".join(lines)


def build_default_assessment() -> BubbleAssessment:
    """
    Build a bubble assessment with known historical reference points.
    Current values will be populated from live data.
    """
    assessment = BubbleAssessment()

    assessment.add(BubbleIndicator(
        name="指数集中度",
        current_value=None,
        dotcom_2000_value=18.0,
        description="前5大股票占标普500总市值的比例。2000年互联网泡沫时微软、思科等占比约18%，集中度越高说明市场越依赖少数巨头。",
        warning_threshold=20.0,
        extreme_threshold=25.0,
        unit="%",
    ))

    assessment.add(BubbleIndicator(
        name="纳斯达克中位市盈率",
        current_value=None,
        dotcom_2000_value=120.0,
        description="纳斯达克所有股票市盈率的中位数。中位数PE越高，说明整体市场越贵。2000年泡沫峰值时达到120倍。",
        warning_threshold=40.0,
        extreme_threshold=80.0,
        unit="倍",
    ))

    assessment.add(BubbleIndicator(
        name="巴菲特指标（总市值/GDP）",
        current_value=None,
        dotcom_2000_value=145.0,
        description="美国股市总市值与美国GDP的比值。巴菲特认为这是衡量市场整体估值最有效的指标。超过150%说明严重高估。",
        warning_threshold=150.0,
        extreme_threshold=180.0,
        unit="%",
    ))

    assessment.add(BubbleIndicator(
        name="AI资本开支/科技巨头营收",
        current_value=None,
        dotcom_2000_value=None,
        description="科技巨头AI基础设施资本开支占其总收入的比例。当资本开支远超营收增长时，可能意味着过度投资。",
        warning_threshold=15.0,
        extreme_threshold=25.0,
        unit="%",
    ))

    assessment.add(BubbleIndicator(
        name="散户情绪指数",
        current_value=None,
        dotcom_2000_value=85.0,
        description="散户投资者的看涨情绪得分（0-100）。当散户普遍看涨时，往往是反向指标。2000年泡沫时散户情绪接近85分。",
        warning_threshold=60.0,
        extreme_threshold=75.0,
        unit="分",
    ))

    assessment.add(BubbleIndicator(
        name="IPO首日涨幅",
        current_value=None,
        dotcom_2000_value=70.0,
        description="新股上市首日平均涨幅。首日涨幅越大，说明市场投机情绪越重。1999年平均首日涨幅达到70%。",
        warning_threshold=30.0,
        extreme_threshold=50.0,
        unit="%",
    ))

    assessment.add(BubbleIndicator(
        name="席勒CAPE（标普500周期调整PE）",
        current_value=None,
        dotcom_2000_value=44.0,
        description="标普500的周期性调整市盈率，使用过去10年平均盈利计算。1999年12月达到历史峰值44倍，这是最著名的估值指标之一。",
        warning_threshold=30.0,
        extreme_threshold=40.0,
        unit="倍",
    ))

    assessment.add(BubbleIndicator(
        name="融资债务/GDP",
        current_value=None,
        dotcom_2000_value=2.5,
        description="券商融资债务（杠杆资金）占GDP的比例。比例越高说明市场杠杆越多，崩盘时的踩踏效应越严重。",
        warning_threshold=3.0,
        extreme_threshold=4.0,
        unit="%",
    ))

    return assessment
