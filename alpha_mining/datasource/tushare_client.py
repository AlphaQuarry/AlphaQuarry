from __future__ import annotations

import enum
import random
import sys
import time
from dataclasses import replace
from datetime import date, datetime
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd


class TushareErrorCategory(enum.Enum):
    """Tushare API 错误分类。"""

    AUTH = "auth"  # 积分不足 / token 无效
    RATE_LIMIT = "rate"  # 频率限制
    NETWORK = "network"  # 网络超时 / 连接失败
    UNKNOWN = "unknown"  # 未分类


def classify_tushare_error(exc: Exception) -> TushareErrorCategory:
    """根据异常类型和消息分类 Tushare API 错误（自适应匹配）。

    实际 Tushare 错误消息示例:
    - 频率限制: "抱歉，您访问接口(xxx)频率超限(1次/分钟)"
    - 权限不足: "抱歉，您没有接口(xxx)访问权限"
    """
    msg = str(exc).lower()
    # 限频相关关键词 (优先级高于 AUTH，因为限频消息也可能含"权限"字样)
    rate_keywords = (
        "频率超限",
        "频次超限",
        "每分钟",
        "频率限制",
        "rate limit",
        "too many",
        "429",
    )
    if any(kw in msg for kw in rate_keywords):
        return TushareErrorCategory.RATE_LIMIT
    # 权限相关关键词 (Tushare 中文消息: "没有接口...访问权限")
    # "没有" + "权限" 组合匹配更精确
    if "没有" in msg and "权限" in msg:
        return TushareErrorCategory.AUTH
    if "积分" in msg:
        return TushareErrorCategory.AUTH
    if any(kw in msg for kw in ("token", "认证", "授权", "permission", "unauthorized")):
        return TushareErrorCategory.AUTH
    # 网络相关
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return TushareErrorCategory.NETWORK
    net_keywords = ("timeout", "connect", "网络", "连接")
    if any(kw in msg for kw in net_keywords):
        return TushareErrorCategory.NETWORK
    return TushareErrorCategory.UNKNOWN


from .config import TushareSettings
from .finance_fields import (
    FINANCE_BALANCESHEET_VIP_FIELDS,
    FINANCE_CASHFLOW_VIP_FIELDS,
    FINANCE_INCOME_VIP_FIELDS,
    FINA_INDICATOR_VIP_FIELDS,
    finance_vip_field_request,
)


_DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
_DAILY_BASIC_FIELDS = "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv"
_ADJ_FACTOR_FIELDS = "ts_code,trade_date,adj_factor"
_STK_LIMIT_FIELDS = "ts_code,trade_date,pre_close,up_limit,down_limit"
_SUSPEND_D_FIELDS = "ts_code,trade_date,suspend_timing,suspend_type"
_NAMECHANGE_FIELDS = "ts_code,name,start_date,end_date,ann_date,change_reason"
_MONEYFLOW_THS_FIELDS = (
    "trade_date,ts_code,name,pct_change,latest,net_amount,net_d5_amount,"
    "buy_lg_amount,buy_lg_amount_rate,buy_md_amount,buy_md_amount_rate,"
    "buy_sm_amount,buy_sm_amount_rate"
)
_MONEYFLOW_FIELDS = (
    "ts_code,trade_date,buy_sm_amount,sell_sm_amount,buy_md_amount,sell_md_amount,"
    "buy_lg_amount,sell_lg_amount,buy_elg_amount,sell_elg_amount,net_mf_amount"
)
_CYQ_PERF_FIELDS = (
    "ts_code,trade_date,his_low,his_high,cost_5pct,cost_15pct,cost_50pct,cost_85pct,cost_95pct,weight_avg,winner_rate"
)
_CYQ_CHIPS_FIELDS = "ts_code,trade_date,price,percent"
_STK_FACTOR_PRO_QFQ_FIELDS = (
    "asi_qfq",
    "asit_qfq",
    "atr_qfq",
    "bbi_qfq",
    "bias1_qfq",
    "bias2_qfq",
    "bias3_qfq",
    "boll_lower_qfq",
    "boll_mid_qfq",
    "boll_upper_qfq",
    "brar_ar_qfq",
    "brar_br_qfq",
    "cci_qfq",
    "cr_qfq",
    "dfma_dif_qfq",
    "dfma_difma_qfq",
    "dmi_adx_qfq",
    "dmi_adxr_qfq",
    "dmi_mdi_qfq",
    "dmi_pdi_qfq",
    "dpo_qfq",
    "madpo_qfq",
    "ema_qfq_5",
    "ema_qfq_10",
    "ema_qfq_20",
    "ema_qfq_30",
    "ema_qfq_60",
    "ema_qfq_90",
    "emv_qfq",
    "maemv_qfq",
    "expma_12_qfq",
    "expma_50_qfq",
    "kdj_qfq",
    "kdj_d_qfq",
    "kdj_k_qfq",
    "ktn_down_qfq",
    "ktn_mid_qfq",
    "ktn_upper_qfq",
    "ma_qfq_20",
    "macd_qfq",
    "macd_dea_qfq",
    "macd_dif_qfq",
    "mass_qfq",
    "ma_mass_qfq",
    "mfi_qfq",
    "mtm_qfq",
    "mtmma_qfq",
    "obv_qfq",
    "psy_qfq",
    "psyma_qfq",
    "roc_qfq",
    "maroc_qfq",
    "rsi_qfq_6",
    "rsi_qfq_24",
    "taq_down_qfq",
    "taq_mid_qfq",
    "taq_up_qfq",
    "trix_qfq",
    "trma_qfq",
    "vr_qfq",
    "wr_qfq",
    "wr1_qfq",
    "xsii_td1_qfq",
    "xsii_td2_qfq",
    "xsii_td3_qfq",
    "xsii_td4_qfq",
)
_STK_FACTOR_PRO_DEFAULT_FIELDS = "ts_code,trade_date," + ",".join(
    [*_STK_FACTOR_PRO_QFQ_FIELDS, "updays", "downdays", "topdays", "lowdays"]
)
_STK_FACTOR_PRO_CHUNK_SIZE = 20
_STK_AUCTION_FIELDS = "ts_code,trade_date,open,high,low,close,vol,amount,vwap"
_REPORT_RC_FIELDS = "ts_code,report_date,org_name,author_name,eps,pe,roe,max_price,min_price,rating,imp_dg"
_INDEX_BASIC_FIELDS = "ts_code,name,market,publisher,category,base_date,base_point,list_date,weight_rule,desc,exp_date"
_INDEX_DAILY_FIELDS = "ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount"
_INDEX_WEIGHT_FIELDS = "index_code,con_code,trade_date,weight"
_INCOME_VIP_FIELDS = finance_vip_field_request(FINANCE_INCOME_VIP_FIELDS)
_BALANCESHEET_VIP_FIELDS = finance_vip_field_request(FINANCE_BALANCESHEET_VIP_FIELDS)
_CASHFLOW_VIP_FIELDS = finance_vip_field_request(FINANCE_CASHFLOW_VIP_FIELDS)
_FINA_INDICATOR_VIP_FIELDS = finance_vip_field_request(FINA_INDICATOR_VIP_FIELDS)
_THS_INDEX_FIELDS = "ts_code,name,count,exchange,list_date,type"
_THS_MEMBER_FIELDS = "ts_code,con_code,con_name,weight,in_date,out_date,is_new"
_DAILY_TRADE_DATE_ONLY_FIELDS = "trade_date"


class TushareClient:
    def __init__(self, settings: TushareSettings):
        token = str(settings.token or "").strip()
        if not token:
            raise ValueError("Tushare token is empty. Set TUSHARE_TOKEN or config.tushare.token")

        try:
            import tushare as ts  # type: ignore
        except Exception as exc:
            raise RuntimeError("tushare is required but not installed") from exc

        self._settings = settings
        self._ts = ts
        # Match stable vendor examples:
        # - pro_api(token) is the primary initialization path
        # - set_token can fail on some locked/home environments, so keep it best-effort
        try:
            ts.set_token(token)
        except Exception:
            pass
        self._pro = ts.pro_api(token)

        # Private deployments with a custom http_url need explicit token and
        # endpoint overrides on the DataApi internals.  Standard Tushare users
        # never reach this branch because http_url defaults to "".
        http_url = _normalize_http_url(str(settings.http_url or "").strip())
        if http_url:
            if hasattr(self._pro, "_DataApi__token"):
                try:
                    setattr(self._pro, "_DataApi__token", token)
                except Exception:
                    pass
            if hasattr(self._pro, "_DataApi__http_url"):
                try:
                    setattr(self._pro, "_DataApi__http_url", http_url)
                except Exception:
                    pass

    @property
    def settings(self) -> TushareSettings:
        return self._settings

    def fetch_trade_cal(
        self,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
        exchange: str = "SSE",
    ) -> pd.DataFrame:
        start_date_norm = _to_tushare_date(start_date)
        end_date_norm = _to_tushare_date(end_date)
        try:
            out = self._call(
                "trade_cal",
                exchange=str(exchange),
                start_date=start_date_norm,
                end_date=end_date_norm,
            )
        except Exception:
            # Some private deployments are stricter on optional params and may reject `exchange`.
            out = self._call(
                "trade_cal",
                start_date=start_date_norm,
                end_date=end_date_norm,
            )
        return _sort_if_present(out, ["cal_date"])

    def fetch_open_trade_dates(
        self,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
        exchange: str = "SSE",
    ) -> list[str]:
        cal = pd.DataFrame()
        try:
            cal = self.fetch_trade_cal(start_date=start_date, end_date=end_date, exchange=exchange)
        except Exception:
            cal = pd.DataFrame()

        dates = _extract_open_dates_from_trade_cal(cal)
        if dates:
            return dates
        return self._probe_open_trade_dates_by_daily(
            start_date=start_date,
            end_date=end_date,
        )

    def fetch_trade_calendar_bundle(
        self,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
        exchange: str = "SSE",
    ) -> tuple[pd.DataFrame, list[str], str]:
        cal = pd.DataFrame()
        open_dates: list[str] = []
        source = "trade_cal"

        try:
            cal = self.fetch_trade_cal(start_date=start_date, end_date=end_date, exchange=exchange)
            open_dates = _extract_open_dates_from_trade_cal(cal)
        except Exception:
            cal = pd.DataFrame()
            open_dates = []

        if not open_dates:
            open_dates = self._probe_open_trade_dates_by_daily(
                start_date=start_date,
                end_date=end_date,
            )
            source = "daily_probe"

        if not open_dates:
            return cal, [], source

        if cal.empty:
            cal = _build_open_only_trade_cal_df(open_dates=open_dates, exchange=str(exchange))
        return cal, open_dates, source

    def fetch_stock_basic(
        self,
        list_statuses: Iterable[str] = ("L", "D", "P"),
        exchange: str = "",
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        fields = "ts_code,symbol,name,area,industry,market,exchange,curr_type,list_status,list_date,delist_date,is_hs"
        for status in list_statuses:
            status_norm = str(status or "").strip().upper()
            if not status_norm:
                continue
            frame = self._call(
                "stock_basic",
                exchange=str(exchange),
                list_status=status_norm,
                fields=fields,
            )
            if frame.empty:
                continue
            frame["list_status"] = status_norm
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(
            subset=[c for c in ["ts_code", "list_status"] if c in out.columns],
            keep="last",
        )
        return _sort_if_present(out, ["ts_code", "list_status"])

    def fetch_index_classify(self, src: str = "SW2021") -> pd.DataFrame:
        out = self._call("index_classify", src=str(src))
        return _sort_if_present(out, ["index_code"])

    def fetch_index_member_all(self, src: str = "SW2021") -> pd.DataFrame:
        out = self._call("index_member_all", src=str(src))
        return _sort_if_present(out, ["con_code", "index_code", "in_date"])

    def fetch_daily_by_trade_date(self, trade_date: str | date | datetime) -> pd.DataFrame:
        out = self._call(
            "daily",
            trade_date=_to_tushare_date(trade_date),
            fields=_DAILY_FIELDS,
        )
        return _sort_if_present(out, ["ts_code"])

    def fetch_daily_basic_by_trade_date(self, trade_date: str | date | datetime) -> pd.DataFrame:
        out = self._call(
            "daily_basic",
            trade_date=_to_tushare_date(trade_date),
            fields=_DAILY_BASIC_FIELDS,
        )
        return _sort_if_present(out, ["ts_code"])

    def fetch_adj_factor_by_trade_date(self, trade_date: str | date | datetime) -> pd.DataFrame:
        out = self._call(
            "adj_factor",
            trade_date=_to_tushare_date(trade_date),
            fields=_ADJ_FACTOR_FIELDS,
        )
        return _sort_if_present(out, ["ts_code"])

    def fetch_stk_limit_by_trade_date(self, trade_date: str | date | datetime) -> pd.DataFrame:
        out = self._call(
            "stk_limit",
            trade_date=_to_tushare_date(trade_date),
            fields=_STK_LIMIT_FIELDS,
        )
        return _sort_if_present(out, ["ts_code"])

    def fetch_suspend_d_by_trade_date(self, trade_date: str | date | datetime) -> pd.DataFrame:
        out = self._call(
            "suspend_d",
            trade_date=_to_tushare_date(trade_date),
            fields=_SUSPEND_D_FIELDS,
        )
        return _sort_if_present(out, ["ts_code"])

    def fetch_namechange(
        self,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        kwargs: dict[str, Any] = {"fields": _NAMECHANGE_FIELDS}
        if start_date is not None and str(start_date).strip():
            kwargs["start_date"] = _to_tushare_date(start_date)
        if end_date is not None and str(end_date).strip():
            kwargs["end_date"] = _to_tushare_date(end_date)
        out = self._call("namechange", **kwargs)
        return _sort_if_present(out, ["ts_code", "start_date", "end_date"])

    def fetch_moneyflow_ths_by_trade_date(self, trade_date: str | date | datetime) -> pd.DataFrame:
        out = self._call(
            "moneyflow_ths",
            trade_date=_to_tushare_date(trade_date),
            fields=_MONEYFLOW_THS_FIELDS,
        )
        return _sort_if_present(out, ["ts_code"])

    def fetch_moneyflow_by_trade_date(self, trade_date: str | date | datetime) -> pd.DataFrame:
        out = self._call(
            "moneyflow",
            trade_date=_to_tushare_date(trade_date),
            fields=_MONEYFLOW_FIELDS,
        )
        return _sort_if_present(out, ["ts_code"])

    def fetch_cyq_perf_by_trade_date(self, trade_date: str | date | datetime) -> pd.DataFrame:
        out = self._call(
            "cyq_perf",
            trade_date=_to_tushare_date(trade_date),
            fields=_CYQ_PERF_FIELDS,
        )
        return _sort_if_present(out, ["ts_code"])

    def fetch_cyq_perf_by_ts_code(
        self,
        ts_code: str,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
    ) -> pd.DataFrame:
        out = self._call(
            "cyq_perf",
            ts_code=str(ts_code or "").strip(),
            start_date=_to_tushare_date(start_date),
            end_date=_to_tushare_date(end_date),
            fields=_CYQ_PERF_FIELDS,
        )
        return _sort_if_present(out, ["ts_code", "trade_date"])

    def fetch_cyq_chips_by_ts_code(
        self,
        ts_code: str,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
    ) -> pd.DataFrame:
        out = self._call(
            "cyq_chips",
            ts_code=str(ts_code or "").strip(),
            start_date=_to_tushare_date(start_date),
            end_date=_to_tushare_date(end_date),
            fields=_CYQ_CHIPS_FIELDS,
        )
        return _sort_if_present(out, ["ts_code", "trade_date", "price"])

    def fetch_stk_factor_pro_by_trade_date(
        self,
        trade_date: str | date | datetime,
        fields: str | None = None,
    ) -> pd.DataFrame:
        trade_date_norm = _to_tushare_date(trade_date)
        fields_norm = str(fields or _STK_FACTOR_PRO_DEFAULT_FIELDS)
        try:
            out = self._call(
                "stk_factor_pro",
                trade_date=trade_date_norm,
                fields=fields_norm,
            )
        except Exception:
            out = self._fetch_stk_factor_pro_by_params_chunked(
                base_kwargs={"trade_date": trade_date_norm},
                fields=fields_norm,
                context=f"trade_date={trade_date_norm}",
            )
        return _sort_if_present(out, ["ts_code"])

    def fetch_stk_factor_pro_by_ts_code(
        self,
        ts_code: str,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
        trade_date: str | date | datetime | None = None,
        fields: str | None = None,
    ) -> pd.DataFrame:
        code = str(ts_code or "").strip()
        if not code:
            raise ValueError("ts_code is required for stk_factor_pro ts_code-range fetch")
        base_kwargs: dict[str, Any] = {"ts_code": code}
        if trade_date is not None and str(trade_date).strip():
            base_kwargs["trade_date"] = _to_tushare_date(trade_date)
        else:
            if start_date is None or end_date is None:
                raise ValueError("start_date and end_date are required when trade_date is not provided")
            base_kwargs["start_date"] = _to_tushare_date(start_date)
            base_kwargs["end_date"] = _to_tushare_date(end_date)

        fields_norm = str(fields).strip() if fields is not None and str(fields).strip() else ""
        call_kwargs = dict(base_kwargs)
        # The stk_factor_pro endpoint is more reliable in ts_code mode when the
        # vendor decides the default output columns. Curation still keeps only
        # the project allowlist, so extra returned fields do not enter the lake.
        if fields_norm:
            call_kwargs["fields"] = fields_norm
        try:
            out = self._call(
                "stk_factor_pro",
                **call_kwargs,
            )
        except Exception:
            context = ",".join(f"{key}={value}" for key, value in base_kwargs.items())
            out = self._fetch_stk_factor_pro_by_params_chunked(
                base_kwargs=base_kwargs,
                fields=fields_norm or _STK_FACTOR_PRO_DEFAULT_FIELDS,
                context=context,
            )
        return _sort_if_present(out, ["ts_code", "trade_date"])

    def _fetch_stk_factor_pro_by_params_chunked(
        self,
        base_kwargs: dict[str, Any],
        fields: str,
        context: str,
    ) -> pd.DataFrame:
        requested = _unique_fields(fields)
        key_fields = [field for field in ("ts_code", "trade_date") if field in requested]
        if not key_fields:
            key_fields = ["ts_code", "trade_date"]
        payload_fields = [field for field in requested if field not in set(key_fields)]
        chunks = _chunks(payload_fields, _STK_FACTOR_PRO_CHUNK_SIZE)
        frames: list[pd.DataFrame] = []
        skipped_fields: list[str] = []
        for chunk in chunks:
            frames.extend(
                self._fetch_stk_factor_pro_field_chunk(
                    base_kwargs=base_kwargs,
                    context=context,
                    key_fields=key_fields,
                    payload_fields=chunk,
                    skipped_fields=skipped_fields,
                )
            )
        if not frames:
            empty = pd.DataFrame(columns=requested)
            empty.attrs["skipped_stk_factor_pro_fields"] = list(skipped_fields)
            return empty
        out = frames[0]
        merge_keys = [field for field in key_fields if field in out.columns]
        for frame in frames[1:]:
            right_keys = [field for field in merge_keys if field in frame.columns]
            if right_keys:
                out = out.merge(frame, on=right_keys, how="outer")
            else:
                out = pd.concat([out, frame], axis=1)
        ordered = [field for field in requested if field in out.columns]
        extras = [field for field in out.columns if field not in ordered]
        result = out[[*ordered, *extras]]
        result.attrs["skipped_stk_factor_pro_fields"] = list(skipped_fields)
        return result

    def _fetch_stk_factor_pro_field_chunk(
        self,
        base_kwargs: dict[str, Any],
        context: str,
        key_fields: list[str],
        payload_fields: list[str],
        skipped_fields: list[str],
    ) -> list[pd.DataFrame]:
        payload = _unique_fields(payload_fields)
        if not payload:
            return []
        chunk_fields = _unique_fields([*key_fields, *payload])
        try:
            frame = self._call(
                "stk_factor_pro",
                **base_kwargs,
                fields=",".join(chunk_fields),
            )
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                return [frame]
            return []
        except Exception as exc:
            if len(payload) <= 1:
                print(
                    f"[tushare_client][warn] stk_factor_pro field skipped "
                    f"{context} field={payload[0]} error={type(exc).__name__}: {exc}"
                )
                if payload[0] not in skipped_fields:
                    skipped_fields.append(payload[0])
                return []
            mid = max(1, len(payload) // 2)
            left = self._fetch_stk_factor_pro_field_chunk(
                base_kwargs=base_kwargs,
                context=context,
                key_fields=key_fields,
                payload_fields=payload[:mid],
                skipped_fields=skipped_fields,
            )
            right = self._fetch_stk_factor_pro_field_chunk(
                base_kwargs=base_kwargs,
                context=context,
                key_fields=key_fields,
                payload_fields=payload[mid:],
                skipped_fields=skipped_fields,
            )
            return [*left, *right]

    def fetch_stk_auction_o_by_trade_date(self, trade_date: str | date | datetime) -> pd.DataFrame:
        out = self._call(
            "stk_auction_o",
            trade_date=_to_tushare_date(trade_date),
            fields=_STK_AUCTION_FIELDS,
        )
        return _sort_if_present(out, ["ts_code"])

    def fetch_stk_auction_c_by_trade_date(self, trade_date: str | date | datetime) -> pd.DataFrame:
        out = self._call(
            "stk_auction_c",
            trade_date=_to_tushare_date(trade_date),
            fields=_STK_AUCTION_FIELDS,
        )
        return _sort_if_present(out, ["ts_code"])

    def fetch_report_rc(
        self,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        kwargs: dict[str, Any] = {"fields": _REPORT_RC_FIELDS}
        if start_date is not None and str(start_date).strip():
            kwargs["start_date"] = _to_tushare_date(start_date)
        if end_date is not None and str(end_date).strip():
            kwargs["end_date"] = _to_tushare_date(end_date)
        out = self._call("report_rc", **kwargs)
        return _sort_if_present(out, ["ts_code", "report_date"])

    def fetch_index_basic(
        self,
        market: str = "",
        publisher: str = "",
        category: str = "",
    ) -> pd.DataFrame:
        kwargs: dict[str, Any] = {"fields": _INDEX_BASIC_FIELDS}
        market_norm = str(market or "").strip()
        publisher_norm = str(publisher or "").strip()
        category_norm = str(category or "").strip()
        if market_norm:
            kwargs["market"] = market_norm
        if publisher_norm:
            kwargs["publisher"] = publisher_norm
        if category_norm:
            kwargs["category"] = category_norm
        out = self._call("index_basic", **kwargs)
        return _sort_if_present(out, ["ts_code"])

    def fetch_index_daily_by_ts_code(
        self,
        ts_code: str,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
    ) -> pd.DataFrame:
        code = str(ts_code or "").strip()
        if not code:
            raise ValueError("ts_code is required for index_daily ts_code-range fetch")
        out = self._call(
            "index_daily",
            ts_code=code,
            start_date=_to_tushare_date(start_date),
            end_date=_to_tushare_date(end_date),
            fields=_INDEX_DAILY_FIELDS,
        )
        return _sort_if_present(out, ["ts_code", "trade_date"])

    def fetch_index_weight_by_index_code(
        self,
        index_code: str,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
    ) -> pd.DataFrame:
        code = str(index_code or "").strip()
        if not code:
            raise ValueError("index_code is required for index_weight fetch")
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
            raise ValueError("index_weight requires a valid start_date/end_date range")

        frames: list[pd.DataFrame] = []
        cur = pd.Timestamp(year=int(start_ts.year), month=int(start_ts.month), day=1)
        end_month = pd.Timestamp(year=int(end_ts.year), month=int(end_ts.month), day=1)
        while cur <= end_month:
            win_start = max(start_ts, cur)
            win_end = min(end_ts, cur + pd.offsets.MonthEnd(0))
            frame = self._call(
                "index_weight",
                index_code=code,
                start_date=_to_tushare_date(win_start),
                end_date=_to_tushare_date(win_end),
                fields=_INDEX_WEIGHT_FIELDS,
            )
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frames.append(frame)
            cur = cur + pd.DateOffset(months=1)
        if not frames:
            return pd.DataFrame(columns=_INDEX_WEIGHT_FIELDS.split(","))
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset=["index_code", "con_code", "trade_date"], keep="last")
        return _sort_if_present(out, ["index_code", "trade_date", "con_code"])

    def fetch_moneyflow_dc_by_trade_date(self, trade_date: str | date | datetime) -> pd.DataFrame:
        _ = trade_date
        raise NotImplementedError("moneyflow_dc is a separate Tushare interface; P3 is wired to moneyflow_ths")

    def fetch_ths_index(self) -> pd.DataFrame:
        out = self._call("ths_index", fields=_THS_INDEX_FIELDS)
        return _sort_if_present(out, ["ts_code"])

    def fetch_ths_member(
        self,
        ts_codes: Iterable[str] | None = None,
        con_code: str | None = None,
    ) -> pd.DataFrame:
        con_code_norm = str(con_code or "").strip()
        if con_code_norm:
            out = self._call("ths_member", con_code=con_code_norm, fields=_THS_MEMBER_FIELDS)
            return _sort_if_present(out, ["ts_code", "code", "in_date"])

        codes = [str(x).strip() for x in (ts_codes or []) if str(x).strip()]
        if not codes:
            index_df = self.fetch_ths_index()
            if "ts_code" in index_df.columns:
                codes = [str(x).strip() for x in index_df["ts_code"].dropna().astype(str).tolist() if str(x).strip()]
        if not codes:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for code in dict.fromkeys(codes):
            frame = self._call("ths_member", ts_code=str(code), fields=_THS_MEMBER_FIELDS)
            if frame is None or frame.empty:
                continue
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(keep="last")
        return _sort_if_present(out, ["ts_code", "code", "in_date"])

    def fetch_income_vip(
        self,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        kwargs: dict[str, Any] = {"fields": _INCOME_VIP_FIELDS}
        if start_date is not None and str(start_date).strip():
            kwargs["start_date"] = _to_tushare_date(start_date)
        if end_date is not None and str(end_date).strip():
            kwargs["end_date"] = _to_tushare_date(end_date)
        out = self._call("income_vip", **kwargs)
        return _sort_if_present(out, ["ts_code", "ann_date", "end_date"])

    def fetch_balancesheet_vip(
        self,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        kwargs: dict[str, Any] = {"fields": _BALANCESHEET_VIP_FIELDS}
        if start_date is not None and str(start_date).strip():
            kwargs["start_date"] = _to_tushare_date(start_date)
        if end_date is not None and str(end_date).strip():
            kwargs["end_date"] = _to_tushare_date(end_date)
        out = self._call("balancesheet_vip", **kwargs)
        return _sort_if_present(out, ["ts_code", "ann_date", "end_date"])

    def fetch_cashflow_vip(
        self,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        kwargs: dict[str, Any] = {"fields": _CASHFLOW_VIP_FIELDS}
        if start_date is not None and str(start_date).strip():
            kwargs["start_date"] = _to_tushare_date(start_date)
        if end_date is not None and str(end_date).strip():
            kwargs["end_date"] = _to_tushare_date(end_date)
        out = self._call("cashflow_vip", **kwargs)
        return _sort_if_present(out, ["ts_code", "ann_date", "end_date"])

    def fetch_fina_indicator_vip(
        self,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        kwargs: dict[str, Any] = {"fields": _FINA_INDICATOR_VIP_FIELDS}
        if start_date is not None and str(start_date).strip():
            kwargs["start_date"] = _to_tushare_date(start_date)
        if end_date is not None and str(end_date).strip():
            kwargs["end_date"] = _to_tushare_date(end_date)
        out = self._call("fina_indicator_vip", **kwargs)
        return _sort_if_present(out, ["ts_code", "ann_date", "end_date"])

    def _probe_open_trade_dates_by_daily(
        self,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
    ) -> list[str]:
        start_date_norm = _to_tushare_date(start_date)
        end_date_norm = _to_tushare_date(end_date)
        probe_codes = ("000001.SZ", "600000.SH")

        for code in probe_codes:
            try:
                out = self._call(
                    "daily",
                    ts_code=str(code),
                    start_date=start_date_norm,
                    end_date=end_date_norm,
                    fields=_DAILY_TRADE_DATE_ONLY_FIELDS,
                )
            except Exception:
                continue
            if out is None or out.empty:
                continue
            if "trade_date" not in out.columns:
                continue
            dates = sorted({str(x) for x in out["trade_date"].dropna().astype(str).tolist() if str(x)})
            if dates:
                return dates
        return []

    def _call(self, api_name: str, **kwargs: Any) -> pd.DataFrame:
        fn = getattr(self._pro, str(api_name))
        attempts = max(1, int(self._settings.max_retries))
        pause = max(0.0, float(self._settings.request_pause_seconds))
        base_sleep = max(0.0, float(self._settings.retry_sleep_seconds))

        last_error: Exception | None = None
        last_category: TushareErrorCategory = TushareErrorCategory.UNKNOWN
        for idx in range(attempts):
            try:
                out = fn(**kwargs)
                if pause > 0:
                    time.sleep(pause)
                if out is None:
                    return pd.DataFrame()
                if not isinstance(out, pd.DataFrame):
                    return pd.DataFrame(out)
                return out
            except Exception as exc:
                last_error = exc
                last_category = classify_tushare_error(exc)
                # 权限错误不重试
                if last_category == TushareErrorCategory.AUTH:
                    break
                if idx >= attempts - 1:
                    break
                # 指数退避 + 抖动: base * 2^idx ± 20%
                delay = base_sleep * (2**idx)
                jitter = delay * 0.2 * (2 * random.random() - 1)
                delay = max(0.1, delay + jitter)
                # 限频错误额外等待
                if last_category == TushareErrorCategory.RATE_LIMIT:
                    delay = max(delay, 5.0)
                time.sleep(delay)

        if last_error is not None:
            error_msg = str(last_error).strip()[:200]
            category_label = last_category.value
            if last_category == TushareErrorCategory.AUTH:
                print(
                    f"\n{'=' * 60}\n"
                    f"[tushare] ❌ 权限不足 api={api_name}\n"
                    f"  错误: {error_msg}\n"
                    f"  原因: 当前 Tushare 账号积分不足，无法访问此接口\n"
                    f"  建议: 升级积分或排除此表 (--exclude-fact-tables {api_name})\n"
                    f"{'=' * 60}\n",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[tushare] ⚠️ api={api_name} category={category_label} error={error_msg}",
                    file=sys.stderr,
                )
            raise TushareApiError(
                f"Tushare API failed: {api_name}({kwargs}) [{category_label}]",
                api_name=api_name,
                category=last_category,
            ) from last_error
        return pd.DataFrame()


class TushareApiError(RuntimeError):
    """Tushare API 调用失败，附带错误分类。"""

    def __init__(
        self,
        message: str,
        *,
        api_name: str = "",
        category: TushareErrorCategory = TushareErrorCategory.UNKNOWN,
    ):
        super().__init__(message)
        self.api_name = api_name
        self.category = category


def build_tushare_client_from_settings(settings: TushareSettings) -> TushareClient:
    return TushareClient(settings=settings)


def override_tushare_token(settings: TushareSettings, token: str) -> TushareSettings:
    return replace(settings, token=str(token or "").strip())


def _to_tushare_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    text = str(value or "").strip()
    if not text:
        raise ValueError("Date value is empty")
    if len(text) == 8 and text.isdigit():
        return text
    try:
        dt = pd.to_datetime(text, errors="raise")
    except Exception as exc:
        raise ValueError(f"Could not parse date: {value}") from exc
    return dt.strftime("%Y%m%d")


def _normalize_http_url(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""

    # Accept plain host names in local/private deploys and default to https.
    if "://" not in text:
        text = "https://" + text

    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "Invalid tushare.http_url. Expected full URL like 'https://api.tushare.pro' or custom endpoint with scheme."
        )
    return text.rstrip("/")


def _sort_if_present(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return df
    return df.sort_values(valid_cols, kind="mergesort").reset_index(drop=True)


def _unique_fields(fields: str | Iterable[str]) -> list[str]:
    raw_values = fields.split(",") if isinstance(fields, str) else list(fields)
    out: list[str] = []
    for raw in raw_values:
        field = str(raw or "").strip()
        if field and field not in out:
            out.append(field)
    return out


def _chunks(values: list[str], size: int) -> list[list[str]]:
    step = max(1, int(size))
    return [values[idx : idx + step] for idx in range(0, len(values), step)] or [[]]


def _extract_open_dates_from_trade_cal(cal: pd.DataFrame) -> list[str]:
    if cal is None or cal.empty:
        return []
    work = cal.copy()
    if "is_open" in work.columns:
        work = work[pd.to_numeric(work["is_open"], errors="coerce") == 1]
    date_col = "cal_date" if "cal_date" in work.columns else ("trade_date" if "trade_date" in work.columns else "")
    if not date_col:
        return []
    dates = sorted({str(x) for x in work[date_col].dropna().astype(str).tolist() if str(x)})
    return dates


def _build_open_only_trade_cal_df(open_dates: list[str], exchange: str) -> pd.DataFrame:
    if not open_dates:
        return pd.DataFrame(columns=["exchange", "cal_date", "is_open", "pretrade_date"])
    normalized = sorted({_to_tushare_date(x) for x in open_dates})
    rows: list[dict[str, Any]] = []
    prev = ""
    for d in normalized:
        rows.append(
            {
                "exchange": str(exchange or ""),
                "cal_date": str(d),
                "is_open": 1,
                "pretrade_date": str(prev),
            }
        )
        prev = str(d)
    return pd.DataFrame(rows)
