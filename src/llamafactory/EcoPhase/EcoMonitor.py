# Copyright 2025 the LlamaFactory team.
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

from __future__ import annotations

import inspect
import types
import os
import time
import threading
import json
import socket
import signal
import subprocess
import urllib.error
import urllib.request
import secrets
from base64 import b64decode as _bd
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist

import grpc
from . import eco_control_pb2, eco_control_pb2_grpc

def _ds(s):
    return _bd(s).decode()


_E0 = _ds("RUNPX0dSUENfQUREUg==")
_E1 = _ds("RUNPX0NMSUVOVF9JRA==")
_E2 = _ds("RUNPX0FQSV9LRVk=")
_E3 = _ds("RUNPX1RMU19ST09UX0NB")
_E4 = _ds("RUNPX1RSQUlOSU5HX0lORk8=")
_E5 = _ds("RUNPX1ZFUlNJT04=")
_E6 = _ds("ZGVmYXVsdF9jbGllbnQ=")
_E7 = _ds("ZGVmYXVsdF9ydW4=")
_E8 = _ds("MC40LjA=")
_E9 = _ds("X2tlcHRfcmF0aW9fbW9uaXRvcg==")


class EcoMonitor:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

        self._a4: int = 0

        self._a7: List = []

        self._a8: Optional[Callable] = None
        self._a9 = None
        self._b0: int = 0

        self._b1: float = 1.0
        self._b2: int = 0
        self._b3: int = 0
        self._b4: bool = False

        self._b5: bool = False
        self._b6: str = ""
        self._b7: str = ""
        self._b8: str = ""
        self._b9: Optional[grpc.Channel] = None
        self._c1: Optional[eco_control_pb2_grpc.EcoControlServiceStub] = None

        self._gateway_url: str = ""
        self._eco_user: str = ""
        self._plugin_version: str = "1.0.0"
        self._device_id: str = ""
        self._request_id: str = ""
        self._session_id: str = ""
        self._session_token: str = ""
        self._session_expires_at: str = ""
        self._session_handshake_expires_at: str = ""
        self._session_status: str = ""
        self._session_ack_reported: bool = False
        self._connect_failed_reported: bool = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop_event = threading.Event()
        self._heartbeat_started: bool = False
        self._abort_session_lock = threading.Lock()
        self._abort_session_inflight: bool = False
        self._abort_session_reported: bool = False
        self._signal_handlers_installed: bool = False
        self._previous_signal_handlers: Dict[int, Any] = {}

        self._c2 = threading.Lock()
        self._c3: Optional[Dict[str, Any]] = None
        self._c4: int = -1

        self._c5: bool = False
        self._c6: float = 1.0

        self._c7: int = 0
        self._c8: bool = False

        self._c9: bool = False
        self._d0: bool = False

        if dist.is_initialized():
            _init_rk = dist.get_rank()
        else:
            try:
                _init_rk = int(os.environ.get("RANK", "0"))
            except ValueError:
                _init_rk = 0
        if _init_rk == 0:
            print(
                "[EcoPhase] ✅ EcoMonitor initialized."
            )


    @classmethod
    def attach(cls, trainer, enabled: bool = True) -> "EcoMonitor":
        required_attrs = ["model", "optimizer", "state", "control", "training_step"]
        missing_attrs = [attr for attr in required_attrs if not hasattr(trainer, attr)]
        if missing_attrs:
            raise ValueError(
                f"Trainer is missing required attributes: {missing_attrs}. "
                f"EcoMonitor requires a trainer with: {required_attrs}"
            )

        _m = cls(enabled=enabled)
        _m._a9 = trainer

        original_training_step = trainer.training_step
        _m._a8 = original_training_step

        try:
            sig = inspect.signature(original_training_step)
            params = list(sig.parameters.keys())
            if params and params[0] == "self":
                params = params[1:]
        except Exception:
            pass

        _m._i0(trainer)
        _m._install_signal_handlers()

        _m._c7 = _m._w0(trainer)

        def wrapped_training_step(self, *args, **kwargs):
            model = None
            if len(args) >= 1:
                model = args[0]
            elif "model" in kwargs:
                model = kwargs["model"]

            is_last_micro_batch = True
            has_accelerator = hasattr(self, 'accelerator') and hasattr(self.accelerator, 'gradient_state')
            if has_accelerator:
                try:
                    is_last_micro_batch = bool(self.accelerator.gradient_state.sync_gradients)
                except Exception:
                    is_last_micro_batch = True

            _m._a4 = self.state.global_step

            if not _m._c8:
                _m._c7 = _m._w0(self)
                _m._c8 = True

            if _m._b0 == 0 and hasattr(self.state, 'max_steps'):
                _m._b0 = self.state.max_steps

            if not _m._b4:
                _m._e0(self)

            if is_last_micro_batch and model is not None and _m.enabled:
                try:
                    _m._r0(model)
                except Exception:
                    pass

            try:
                result = _m._a8(*args, **kwargs)
            except KeyboardInterrupt:
                _m._report_abort_session("user_interrupted", "KeyboardInterrupt")
                raise
            except Exception as e:
                _m._report_abort_session("training_exception", str(e))
                raise

            if is_last_micro_batch and _m.enabled:
                try:
                    _m._p0(self)
                except Exception:
                    pass

            _m._x0()

            return result

        trainer.training_step = types.MethodType(wrapped_training_step, trainer)

        setattr(trainer, _E9, _m)

        return _m

    def detach(self) -> None:
        if self._a9 is None:
            return

        if self._a8 is not None:
            self._a9.training_step = types.MethodType(
                self._a8, self._a9
            )

        self._stop_heartbeat_thread()
        self._restore_signal_handlers()
        self._x0()

        if hasattr(self._a9, _E9):
            delattr(self._a9, _E9)

    def __del__(self):
        try:
            self._stop_heartbeat_thread()
        except Exception:
            pass
        try:
            self._restore_signal_handlers()
        except Exception:
            pass

    # ========== Internal Methods ==========

    def _install_signal_handlers(self) -> None:
        if self._signal_handlers_installed:
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous_signal_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_termination_signal)
            except Exception as exc:
                print(f"[EcoPhase] Failed to install signal handler for {sig}: {exc}")
        self._signal_handlers_installed = True

    def _restore_signal_handlers(self) -> None:
        if not self._signal_handlers_installed:
            return
        for sig, previous_handler in list(self._previous_signal_handlers.items()):
            try:
                if signal.getsignal(sig) == self._handle_termination_signal:
                    signal.signal(sig, previous_handler)
            except Exception as exc:
                print(f"[EcoPhase] Failed to restore signal handler for {sig}: {exc}")
        self._previous_signal_handlers.clear()
        self._signal_handlers_installed = False

    def _handle_termination_signal(self, signum, frame) -> None:
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        try:
            self._report_abort_session("user_interrupted", signal_name)
        finally:
            previous_handler = self._previous_signal_handlers.get(signum)
            if callable(previous_handler) and previous_handler != self._handle_termination_signal:
                previous_handler(signum, frame)
                return
            if signum == signal.SIGINT:
                raise KeyboardInterrupt()
            raise SystemExit(128 + int(signum))

    @staticmethod
    def _mask_secret(value: str) -> str:
        value = str(value or "")
        if len(value) <= 8:
            return "***"
        return f"{value[:4]}...{value[-4:]}"

    @staticmethod
    def _normalize_http_base_url(value: str) -> str:
        value = str(value or "").strip().rstrip("/")
        if not value:
            return ""
        if value.startswith(("http://", "https://")):
            return value
        return "http://" + value

    @staticmethod
    def _is_http_url(value: str) -> bool:
        return str(value or "").strip().startswith(("http://", "https://"))

    def _build_request_id(self) -> str:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"plugin-start-{timestamp}-{secrets.token_hex(4)}"

    def _float_env(self, name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)) or default)
        except (TypeError, ValueError):
            return float(default)

    def _summarize_device_names(self, names: List[str]) -> tuple[int, str]:
        counts: Dict[str, int] = {}
        order: List[str] = []
        for raw_name in names:
            name = str(raw_name or "unknown").strip() or "unknown"
            if name not in counts:
                counts[name] = 0
                order.append(name)
            counts[name] += 1

        device_count = len(names)
        if device_count <= 0:
            return 0, "unknown"
        if len(order) == 1:
            return device_count, order[0]
        return device_count, "; ".join(f"{name} x{counts[name]}" for name in order)

    def _collect_cuda_type_summary(self) -> tuple[int, str]:
        try:
            if not torch.cuda.is_available():
                return 0, "unknown"
            device_count = int(torch.cuda.device_count() or 0)
            names = []
            for device_idx in range(device_count):
                try:
                    names.append(str(torch.cuda.get_device_name(device_idx) or "unknown"))
                except Exception:
                    names.append("unknown")
            return self._summarize_device_names(names)
        except Exception:
            return 0, "unknown"

    def _collect_torch_npu_type_summary(self) -> tuple[int, str]:
        try:
            npu = getattr(torch, "npu", None)
            if npu is None:
                try:
                    __import__("torch_npu")
                    npu = getattr(torch, "npu", None)
                except Exception:
                    return 0, "unknown"
            if npu is None or not callable(getattr(npu, "is_available", None)) or not npu.is_available():
                return 0, "unknown"

            device_count = int(npu.device_count() or 0)
            names = []
            for device_idx in range(device_count):
                name = ""
                try:
                    get_device_name = getattr(npu, "get_device_name", None)
                    if callable(get_device_name):
                        name = str(get_device_name(device_idx) or "")
                except Exception:
                    name = ""
                if not name:
                    try:
                        props = npu.get_device_properties(device_idx)
                        name = str(getattr(props, "name", "") or "")
                    except Exception:
                        name = ""
                names.append(name or "Ascend NPU")
            return self._summarize_device_names(names)
        except Exception:
            return 0, "unknown"

    def _collect_mtt_type_summary(self) -> tuple[int, str]:
        try:
            musa = getattr(torch, "musa", None)
            if musa is None:
                try:
                    __import__("torch_musa")
                    musa = getattr(torch, "musa", None)
                except Exception:
                    musa = None
            if musa is not None and callable(getattr(musa, "is_available", None)) and musa.is_available():
                device_count = int(musa.device_count() or 0)
                names = []
                for device_idx in range(device_count):
                    name = ""
                    try:
                        get_device_name = getattr(musa, "get_device_name", None)
                        if callable(get_device_name):
                            name = str(get_device_name(device_idx) or "")
                    except Exception:
                        name = ""
                    if not name:
                        try:
                            props = musa.get_device_properties(device_idx)
                            name = str(getattr(props, "name", "") or "")
                        except Exception:
                            name = ""
                    names.append(name or "MTT MUSA")
                return self._summarize_device_names(names)
        except Exception:
            pass

        for command in (["mthreads-gmi", "-L"], ["musa-smi", "-L"], ["mt-smi", "-L"]):
            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=2,
                )
                if result.returncode != 0:
                    continue
                names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if names:
                    return self._summarize_device_names(names)
            except Exception:
                continue
        return 0, "unknown"

    def _collect_xpu_type_summary(self) -> tuple[int, str]:
        try:
            xpu = getattr(torch, "xpu", None)
            if xpu is None:
                try:
                    __import__("intel_extension_for_pytorch")
                    xpu = getattr(torch, "xpu", None)
                except Exception:
                    return 0, "unknown"
            if xpu is None or not callable(getattr(xpu, "is_available", None)) or not xpu.is_available():
                return 0, "unknown"

            device_count = int(xpu.device_count() or 0)
            names = []
            for device_idx in range(device_count):
                name = ""
                try:
                    get_device_name = getattr(xpu, "get_device_name", None)
                    if callable(get_device_name):
                        name = str(get_device_name(device_idx) or "")
                except Exception:
                    name = ""
                if not name:
                    try:
                        props = xpu.get_device_properties(device_idx)
                        name = str(getattr(props, "name", "") or "")
                    except Exception:
                        name = ""
                names.append(name or "Intel XPU")
            return self._summarize_device_names(names)
        except Exception:
            return 0, "unknown"

    def _collect_mlu_type_summary(self) -> tuple[int, str]:
        try:
            mlu = getattr(torch, "mlu", None)
            if mlu is None:
                try:
                    __import__("torch_mlu")
                    mlu = getattr(torch, "mlu", None)
                except Exception:
                    return 0, "unknown"
            if mlu is None or not callable(getattr(mlu, "is_available", None)) or not mlu.is_available():
                return 0, "unknown"

            device_count = int(mlu.device_count() or 0)
            names = []
            for device_idx in range(device_count):
                name = ""
                try:
                    get_device_name = getattr(mlu, "get_device_name", None)
                    if callable(get_device_name):
                        name = str(get_device_name(device_idx) or "")
                except Exception:
                    name = ""
                if not name:
                    try:
                        props = mlu.get_device_properties(device_idx)
                        name = str(getattr(props, "name", "") or "")
                    except Exception:
                        name = ""
                names.append(name or "Cambricon MLU")
            return self._summarize_device_names(names)
        except Exception:
            return 0, "unknown"

    def _collect_gpu_type_summary(self) -> tuple[int, str]:
        for collector in (
            self._collect_cuda_type_summary,
            self._collect_torch_npu_type_summary,
            self._collect_mtt_type_summary,
            self._collect_xpu_type_summary,
            self._collect_mlu_type_summary,
        ):
            try:
                device_count, device_type = collector()
                if int(device_count or 0) > 0:
                    return int(device_count), str(device_type or "unknown")
            except Exception:
                continue
        return 0, "unknown"

    def _resolve_primary_ip(self) -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                ip_addr = sock.getsockname()[0].strip()
                if ip_addr:
                    return ip_addr
        except Exception:
            pass

        try:
            hostname = socket.gethostname().strip()
            ip_addr = socket.gethostbyname(hostname).strip()
            if ip_addr:
                return ip_addr
        except Exception:
            pass

        return "unknown-ip"

    def _resolve_device_id(self) -> str:
        hostname = ""
        for env_name in ("HOSTNAME", "COMPUTERNAME"):
            value = os.environ.get(env_name, "").strip()
            if value:
                hostname = value
                break

        if not hostname:
            try:
                hostname = socket.gethostname().strip()
            except Exception:
                hostname = ""

        return f"{hostname or 'unknown-host'}-{self._resolve_primary_ip()}"

    def _post_json(self, base_url: str, path: str, payload: Dict[str, Any], timeout: float) -> tuple[int, Any]:
        endpoint = base_url.rstrip("/") + path
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Request-Id": str(payload.get("request_id", "")),
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
                return response.status, json.loads(data.decode("utf-8")) if data else {}
        except urllib.error.HTTPError as exc:
            data = exc.read()
            try:
                parsed: Any = json.loads(data.decode("utf-8")) if data else {}
            except json.JSONDecodeError:
                parsed = data.decode("utf-8", errors="replace")
            return exc.code, parsed

    def _report_session_ack(
        self,
        status_value: str,
        session_id: str = "",
        reason: str = "",
        message: str = "",
    ) -> bool:
        if not self._gateway_url or not self._request_id or self._session_ack_reported:
            return False

        status_value = str(status_value or "").strip()
        if status_value not in ("received", "response_lost", "response_invalid"):
            status_value = "response_invalid"

        safe_message = str(message or "")
        api_key = os.environ.get(_E2, "").strip()
        for secret_value in (api_key, self._session_token):
            if secret_value:
                safe_message = safe_message.replace(secret_value, "[secret]")
        safe_message = safe_message[:500]

        payload = {
            "request_id": self._request_id,
            "status": status_value,
            "reason": str(reason or ""),
            "message": safe_message,
        }
        if session_id:
            payload["session_id"] = str(session_id)

        timeout = self._float_env("ECO_GATEWAY_TIMEOUT", 10.0)
        try:
            ack_status, response = self._post_json(
                self._gateway_url,
                "/gateway/v1/ecotrain/sessions/ack",
                payload,
                timeout,
            )
            self._session_ack_reported = True
            print(
                "[EcoPhase] Reported session ack: "
                f"request_id={self._request_id}, status={status_value}, "
                f"session_id={session_id}, http_status={ack_status}, response={response!r}"
            )
            return 200 <= int(ack_status) < 300
        except Exception as exc:
            print(
                "[EcoPhase] Failed to report session ack: "
                f"request_id={self._request_id}, status={status_value}, reason={reason}, error={exc}"
            )
            return False

    def _report_ecotrain_connect_failed(self, reason: str, message: str = "") -> None:
        if not self._gateway_url or not self._session_id or self._connect_failed_reported:
            return

        self._connect_failed_reported = True
        safe_message = str(message or "")
        if self._session_token:
            safe_message = safe_message.replace(self._session_token, "[session_token]")
        safe_message = safe_message[:500]
        payload = {
            "request_id": self._request_id,
            "reason": str(reason or "grpc_connect_failed"),
            "message": safe_message,
        }
        timeout = self._float_env("ECO_GATEWAY_TIMEOUT", 10.0)
        try:
            status, response = self._post_json(
                self._gateway_url,
                f"/gateway/v1/ecotrain/sessions/{self._session_id}/connect-failed",
                payload,
                timeout,
            )
            print(
                "[EcoPhase] Reported EcoTrain connect-failed: "
                f"session_id={self._session_id}, status={status}, response={response!r}"
            )
        except Exception as exc:
            print(
                "[EcoPhase] Failed to report connect-failed: "
                f"session_id={self._session_id}, reason={payload['reason']}, error={exc}"
            )

    def _heartbeat_interval_seconds(self) -> float:
        return max(1.0, self._float_env("ECO_HEARTBEAT_INTERVAL_SEC", 60.0))

    def _heartbeat_timestamp(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _send_heartbeat_once(self) -> bool:
        if not self.enabled or not self._c1:
            return False
        if not self._session_id or not self._session_token:
            return False
        request = eco_control_pb2.HeartbeatRequest(
            session_id=self._session_id,
            session_token=self._session_token,
            request_id=self._request_id,
            run_id=self._b8 or self._request_id,
            timestamp=self._heartbeat_timestamp(),
        )
        timeout = self._float_env("ECO_HEARTBEAT_TIMEOUT_SEC", 10.0)
        try:
            reply = self._c1.Heartbeat(request, timeout=timeout)
            if not getattr(reply, "ok", False):
                print(
                    "[EcoPhase] Heartbeat rejected: "
                    f"session_id={self._session_id}, reason={getattr(reply, 'reason', '')}"
                )
                return False
            return True
        except Exception as exc:
            print(f"[EcoPhase] Heartbeat failed: session_id={self._session_id}, error={exc}")
            return False

    def _heartbeat_loop(self) -> None:
        interval = self._heartbeat_interval_seconds()
        while not self._heartbeat_stop_event.wait(interval):
            self._send_heartbeat_once()

    def _start_heartbeat_thread(self) -> None:
        if self._heartbeat_started:
            return
        if not self._gateway_url or not self._session_id or not self._session_token:
            return
        if not self._c1:
            return
        self._heartbeat_stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="EcoMonitorHeartbeat",
            daemon=True,
        )
        self._heartbeat_started = True
        self._heartbeat_thread.start()

    def _stop_heartbeat_thread(self) -> None:
        self._heartbeat_stop_event.set()
        thread = self._heartbeat_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._heartbeat_thread = None
        self._heartbeat_started = False

    def _report_abort_session(self, reason: str, message: str = "") -> bool:
        with self._abort_session_lock:
            if self._abort_session_reported or self._abort_session_inflight:
                return False
            if not self._c1 or not self._session_id or not self._session_token:
                return False
            self._abort_session_inflight = True

        timeout = self._float_env("ECO_ABORT_TIMEOUT_SEC", 10.0)
        try:
            request = eco_control_pb2.AbortSessionRequest(
                session_id=self._session_id,
                session_token=self._session_token,
                request_id=self._request_id,
                run_id=self._b8 or self._request_id,
                timestamp=self._heartbeat_timestamp(),
                reason=str(reason or "user_interrupted"),
                message=str(message or "")[:500],
                current_step=int(self._a4 or 0),
                total_steps=int(self._b0 or 0),
                ended_at=self._heartbeat_timestamp(),
            )
            reply = self._c1.AbortSession(request, timeout=timeout)
            accepted = bool(getattr(reply, "accepted", False))
            if accepted:
                with self._abort_session_lock:
                    self._abort_session_reported = True
            print(
                "[EcoPhase] AbortSession reported: "
                f"session_id={self._session_id}, accepted={accepted}, "
                f"reason={getattr(reply, 'reason', '')}"
            )
            return accepted
        except Exception as exc:
            print(f"[EcoPhase] AbortSession failed: session_id={self._session_id}, reason={reason}, error={exc}")
            return False
        finally:
            with self._abort_session_lock:
                self._abort_session_inflight = False

    def _create_ecotrain_session(self) -> bool:
        if not self._gateway_url:
            return True

        api_key = os.environ.get(_E2, "").strip()
        if not self._eco_user:
            print("[EcoPhase] 🛑 ECO_CLIENT_ID is required to create EcoTrain session.")
            return False
        if not api_key:
            print("[EcoPhase] 🛑 ECO_API_KEY is required to create EcoTrain session.")
            return False

        payload = {
            "user": self._eco_user,
            "api_key": api_key,
            "plugin_type": "training_acceleration",
            "plugin_version": self._plugin_version,
            "device_id": self._device_id,
            "request_id": self._request_id,
        }

        timeout = self._float_env("ECO_GATEWAY_TIMEOUT", 10.0)
        print(
            "[EcoPhase] Creating EcoTrain session: "
            f"gateway={self._gateway_url}, user={self._eco_user}, "
            f"plugin_version={self._plugin_version}, device_id={self._device_id}, "
            f"request_id={self._request_id}, api_key={self._mask_secret(api_key)}"
        )

        try:
            status, response = self._post_json(
                self._gateway_url,
                "/gateway/v1/ecotrain/sessions",
                payload,
                timeout,
            )
        except Exception as exc:
            print(f"[EcoPhase] 🛑 Failed to create EcoTrain session: {exc}")
            self._report_session_ack(
                "response_lost",
                reason=exc.__class__.__name__,
                message="plugin did not receive create-session response: " + str(exc),
            )
            return False

        if not isinstance(response, dict):
            print(f"[EcoPhase] 🛑 Invalid EcoTrain session response: {response!r}")
            self._report_session_ack(
                "response_invalid",
                reason="invalid_response_type",
                message=f"create-session response is not a JSON object: {type(response).__name__}",
            )
            return False

        if status < 200 or status >= 300 or not response.get("allowed", False):
            print(
                "[EcoPhase] 🛑 EcoTrain session rejected: "
                f"status={status}, error_code={response.get('error_code', '')}"
            )
            return False

        session_id = str(response.get("session_id", "")).strip()
        session_token = str(response.get("session_token", "")).strip()
        host = str(response.get("ecotrain_host", "")).strip()
        port = response.get("ecotrain_port", 0)
        try:
            port_int = int(port)
        except (TypeError, ValueError):
            port_int = 0

        if not session_id or not session_token or not host or port_int <= 0:
            print(
                "[EcoPhase] 🛑 EcoTrain session response missing required fields: "
                f"session_id={bool(session_id)}, token={bool(session_token)}, "
                f"host={bool(host)}, port={port}"
            )
            missing = []
            if not session_id:
                missing.append("session_id")
            if not session_token:
                missing.append("session_token")
            if not host:
                missing.append("ecotrain_host")
            if port_int <= 0:
                missing.append("ecotrain_port")
            self._report_session_ack(
                "response_invalid",
                session_id=session_id,
                reason="missing_required_field",
                message="missing required field: " + ",".join(missing),
            )
            return False

        self._session_id = session_id
        self._session_token = session_token
        self._session_expires_at = str(response.get("expires_at", ""))
        self._session_handshake_expires_at = str(response.get("handshake_expires_at", ""))
        self._session_status = str(response.get("status", ""))
        self._b6 = f"{host}:{port_int}"

        print(
            "[EcoPhase] ✅ EcoTrain session created: "
            f"session_id={self._session_id}, "
            f"session_token={self._mask_secret(self._session_token)}, "
            f"grpc_addr={self._b6}, status={self._session_status}"
        )
        self._report_session_ack("received", session_id=self._session_id)
        return True

    def _w0(self, trainer) -> int:
        args = getattr(trainer, "args", None)
        if args is None:
            return 0

        warmup_steps_cfg = getattr(args, 'warmup_steps', None)
        if warmup_steps_cfg is not None and warmup_steps_cfg > 0:
            return warmup_steps_cfg

        warmup_ratio = getattr(args, 'warmup_ratio', None)
        if warmup_ratio is not None and warmup_ratio > 0:
            args_max_steps = getattr(args, 'max_steps', None)
            if args_max_steps is not None and args_max_steps > 0:
                return int(args_max_steps * warmup_ratio)

            if hasattr(trainer, 'state') and hasattr(trainer.state, 'max_steps'):
                max_steps = trainer.state.max_steps
                if max_steps is not None and max_steps > 0:
                    return int(max_steps * warmup_ratio)

            return 0

        return 0

    def _e0(self, trainer) -> None:
        args = getattr(trainer, "args", None)
        if args is None:
            return

        if hasattr(args, 'num_train_epochs') and args.num_train_epochs > 0:
            self._b1 = float(args.num_train_epochs)
        elif hasattr(trainer, 'state') and hasattr(trainer.state, 'num_train_epochs'):
            self._b1 = float(trainer.state.num_train_epochs)

        try:
            train_dataset = getattr(trainer, 'train_dataset', None)
            if train_dataset is not None:
                if hasattr(train_dataset, '__len__'):
                    self._b2 = len(train_dataset)
                elif hasattr(train_dataset, 'num_rows'):
                    self._b2 = train_dataset.num_rows
        except Exception:
            pass

        if self._b2 > 0 and self._b1 > 0:
            self._b3 = int(self._b2 * self._b1)

        if self._b1 > 0 or self._b2 > 0:
            self._b4 = True

    # ---- gRPC ----

    def _i0(self, trainer) -> None:
        self._b7 = os.environ.get(_E1, _E6)
        raw_addr = os.environ.get(_E0, "").strip()
        configured_gateway_url = os.environ.get("ECO_GATEWAY_URL", "").strip()

        self._gateway_url = (
            self._normalize_http_base_url(configured_gateway_url)
            if configured_gateway_url
            else ""
        )

        if not self._gateway_url and self._is_http_url(raw_addr):
            self._gateway_url = self._normalize_http_base_url(raw_addr)
            self._b6 = ""
        else:
            self._b6 = raw_addr

        self._eco_user = os.environ.get("ECO_USER", "").strip() or self._b7
        self._plugin_version = os.environ.get("ECO_PLUGIN_VERSION", "1.0.0").strip() or "1.0.0"
        self._device_id = self._resolve_device_id()
        self._request_id = self._build_request_id()

        args = getattr(trainer, "args", None)
        self._b8 = self._request_id

        if hasattr(trainer, 'state') and hasattr(trainer.state, 'max_steps'):
            self._b0 = trainer.state.max_steps
        elif args is not None and hasattr(args, 'max_steps') and args.max_steps > 0:
            self._b0 = args.max_steps
        else:
            self._b0 = 0

        self._e0(trainer)

        if self._gateway_url:
            if not self._create_ecotrain_session():
                self._b5 = False
                self._b9 = None
                self._c1 = None
                if not self._d0:
                    self._d0 = True
                    print("[EcoPhase] 🛑 API is disabled.")
                return

        if not self._b6:
            self._b5 = False
            if not self._d0:
                self._d0 = True
                print("[EcoPhase] 🛑 API is disabled.")
            return

        try:
            root_ca_path = os.environ.get(_E3, "").strip()
            if root_ca_path and os.path.exists(root_ca_path):
                with open(root_ca_path, "rb") as f:
                    root_ca = f.read()
                creds = grpc.ssl_channel_credentials(root_certificates=root_ca)
                self._b9 = grpc.secure_channel(self._b6, creds)
            else:
                self._b9 = grpc.insecure_channel(self._b6)

            self._c1 = eco_control_pb2_grpc.EcoControlServiceStub(self._b9)
            if self._gateway_url:
                connect_timeout = self._float_env("ECO_GRPC_CONNECT_TIMEOUT", 5.0)
                grpc.channel_ready_future(self._b9).result(timeout=connect_timeout)
            self._b5 = self.enabled and (self._c1 is not None)
            self._start_heartbeat_thread()
        except Exception as exc:
            if self._gateway_url and self._session_id:
                reason = "dial_timeout" if isinstance(exc, grpc.FutureTimeoutError) else "grpc_connect_failed"
                self._report_ecotrain_connect_failed(reason, str(exc))
            self._stop_heartbeat_thread()
            self._b5 = False
            self._b9 = None
            self._c1 = None
            if not self._d0:
                self._d0 = True
                print("[EcoPhase] 🛑 API is disabled.")

    def _s0(self, step: int, grad_norms: torch.Tensor, current_lr: float,
            obfuscation_data: Dict[str, torch.Tensor]) -> None:
        if not self.enabled:
            return
        if self._c1 is None:
            return

        try:
            if grad_norms.dim() == 1:
                grad_norms_2d = grad_norms.view(1, -1)
            elif grad_norms.dim() == 2:
                grad_norms_2d = grad_norms
            else:
                grad_norms_2d = grad_norms.view(1, -1)

            grad_l1_norms = obfuscation_data['grad_l1_norms'].tolist()
            grad_l2_norms = obfuscation_data['grad_l2_norms'].tolist()
            grad_means_list = obfuscation_data['grad_means'].tolist()
            grad_stds_list = obfuscation_data['grad_stds'].tolist()
            grad_mins_list = obfuscation_data['grad_mins'].tolist()
            grad_maxs_list = obfuscation_data['grad_maxs'].tolist()
            sampled_values = obfuscation_data['sampled_values'].tolist()

            rows = [row.tolist() for row in grad_norms_2d]

        except Exception:
            return

        client_id = self._b7 or _E6
        run_id = self._b8 or _E7
        step_int = int(step)
        lr_val = float(current_lr)
        api_key = os.environ.get(_E2, "").strip()

        training_info = os.environ.get(_E4, "").strip()
        version = os.environ.get(_E5, _E8).strip()

        if 'num_epochs=' not in training_info:
            epoch_parts = []

            if self._b1 > 0:
                epoch_parts.append(f"num_epochs={self._b1:.4f}")

            if self._b2 > 0:
                epoch_parts.append(f"samples_per_epoch={self._b2}")

            if self._b3 > 0:
                epoch_parts.append(f"total_train_samples={self._b3}")

            if epoch_parts:
                epoch_info_str = ",".join(epoch_parts)
                if training_info:
                    training_info = f"{training_info},{epoch_info_str}"
                else:
                    training_info = epoch_info_str

        num_gpus = 0
        gpu_type = "unknown"
        world_size = 1
        rank = 0
        batch_size = 0
        gradient_accumulation_steps = 0
        effective_batch_size = 0
        model_name = "unknown"

        try:
            if torch.cuda.is_available():
                num_gpus, gpu_type = self._collect_gpu_type_summary()

            if dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()

            if self._a9:
                args = getattr(self._a9, 'args', None)
                if args:
                    batch_size = getattr(args, 'per_device_train_batch_size', 0)
                    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
                    effective_batch_size = batch_size * gradient_accumulation_steps * world_size

                    model_path = getattr(args, 'model_name_or_path', '')
                    if model_path:
                        model_name = os.path.basename(model_path.rstrip('/'))

        except Exception:
            pass

        def _wk():
            try:
                request = eco_control_pb2.ControlRequest(
                    client_id=client_id,
                    run_id=run_id,
                    step=step_int,
                    lr=lr_val,
                    total_steps=self._b0,
                    trainer_warmup_steps=self._c7,
                    training_info=training_info,
                    version=version,
                    num_gpus=num_gpus,
                    gpu_type=gpu_type,
                    world_size=world_size,
                    rank=rank,
                    batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    effective_batch_size=effective_batch_size,
                    model_name=model_name,
                )
                if api_key:
                    request.api_key = api_key
                if self._session_id:
                    request.session_id = self._session_id
                if self._session_token:
                    request.session_token = self._session_token
                if self._request_id:
                    request.request_id = self._request_id

                for row in rows:
                    grad_row = eco_control_pb2.GradRow()
                    grad_row.values.extend(row)
                    request.grad_norms.append(grad_row)

                request.grad_l1_norms.extend(grad_l1_norms)
                request.grad_l2_norms.extend(grad_l2_norms)
                request.grad_means.extend(grad_means_list)
                request.grad_stds.extend(grad_stds_list)
                request.sampled_values.extend(sampled_values)

                reply = self._c1.ControlStep(request, timeout=2.0)

                new_lr_val = float(getattr(reply, "new_lr", 0.0))
                lr_multiplier_val = float(getattr(reply, "lr_multiplier", 0.0))
                should_stop_val = bool(getattr(reply, "should_stop", False))
                step_reply = int(getattr(reply, "step", step_int))

                stop_summary = str(getattr(reply, "stop_summary", ""))

                control = {
                    "step": step_reply,
                    "new_lr": new_lr_val if new_lr_val >= 0 else None,
                    "lr_multiplier": lr_multiplier_val if lr_multiplier_val >= 0 else None,
                    "should_stop": should_stop_val,
                    "stop_summary": stop_summary,
                }

                with self._c2:
                    self._c3 = control

                if not self._c9:
                    self._c9 = True
                    print("[EcoPhase] ✅ API is enabled.")

            except Exception:
                if not self._c9 and not self._d0:
                    self._d0 = True
                    print("[EcoPhase] 🛑 API is disabled.")

        threading.Thread(
            target=_wk,
            name=f"ecs_{step_int}",
            daemon=True,
        ).start()

    # ---- Hook ----

    def _r0(self, model) -> None:
        unwrapped_model = model
        if hasattr(model, "module"):
            unwrapped_model = model.module

        if hasattr(unwrapped_model, "base_model") and hasattr(
            unwrapped_model.base_model, "model"
        ):
            base_model = unwrapped_model.base_model.model
        else:
            base_model = unwrapped_model

        lm_head = None

        for attr_name in ["lm_head", "output_layer", "output", "classifier", "score", "head"]:
            if hasattr(base_model, attr_name):
                lm_head = getattr(base_model, attr_name)
                break

        if lm_head is None and hasattr(base_model, "transformer"):
            transformer = base_model.transformer
            for attr_name in ["output_layer", "lm_head"]:
                if hasattr(transformer, attr_name):
                    lm_head = getattr(transformer, attr_name)
                    break

        if lm_head is None:
            return

        handle = lm_head.register_full_backward_hook(self._h0)
        self._a7.append(handle)

    def _h0(self, module, grad_input, grad_output):
        try:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank != 0:
                return None

            if not self._b5 or self._c1 is None:
                return None

            if isinstance(grad_output, (list, tuple)):
                grad = grad_output[0]
            else:
                grad = grad_output

            if grad is None or not isinstance(grad, torch.Tensor):
                return None

            if grad.dim() != 3:
                return None

            batch_size, seq_len, vocab_size = grad.shape
            grad_flat = grad.reshape(-1, vocab_size)

            grad_l1_norms = torch.sum(torch.abs(grad_flat), dim=1)

            grad_squared = grad_flat * grad_flat
            sum_squared = torch.sum(grad_squared, dim=1)
            grad_l2_norms = torch.sqrt(sum_squared)

            grad_means = torch.mean(grad_flat, dim=1)
            grad_stds = torch.std(grad_flat, dim=1)

            grad_mins = torch.min(grad_flat, dim=1)[0]
            grad_maxs = torch.max(grad_flat, dim=1)[0]

            sampled_values = torch.tensor([], dtype=torch.float32)

            obfuscation_data = {
                'grad_l1_norms': grad_l1_norms.detach().cpu(),
                'grad_l2_norms': grad_l2_norms.detach().cpu(),
                'grad_means': grad_means.detach().cpu(),
                'grad_stds': grad_stds.detach().cpu(),
                'grad_mins': grad_mins.detach().cpu(),
                'grad_maxs': grad_maxs.detach().cpu(),
                'sampled_values': sampled_values,
            }

            norms_2d = grad_l2_norms.view(batch_size, seq_len)

            current_lr = 0.0
            if (
                self._a9 is not None
                and self._a9.optimizer is not None
                and self._a9.optimizer.param_groups
            ):
                current_lr = float(self._a9.optimizer.param_groups[0]["lr"])

            step = int(self._a4)
            norms_cpu = norms_2d.detach().cpu()

            self._s0(
                step=step,
                grad_norms=norms_cpu,
                current_lr=current_lr,
                obfuscation_data=obfuscation_data,
            )

        except Exception:
            pass

        return None

    # ---- Control ----

    def _p0(self, trainer) -> None:
        if not self._b5:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        should_stop = False
        new_lr_from_server: Optional[float] = None
        lr_multiplier: Optional[float] = None
        use_fallback = False

        control_to_apply: Optional[Dict[str, Any]] = None

        if rank == 0:
            with self._c2:
                control = (
                    self._c3.copy()
                    if self._c3 is not None
                    else None
                )

            current_step = self._a4

            if control is not None:
                control_step = int(control.get("step", -1))

                if control_step == current_step:
                    control_to_apply = control
                    self._c4 = control_step
                else:
                    control_to_apply = None
            else:
                control_to_apply = None

            if control_to_apply is not None:
                should_stop = bool(control_to_apply.get("should_stop", False))
                new_lr_from_server = control_to_apply.get("new_lr", None)
                lr_multiplier = control_to_apply.get("lr_multiplier", None)

                if lr_multiplier is not None:
                    self._c6 = lr_multiplier
                    if not self._c5:
                        self._c5 = True
            elif self._c5:
                use_fallback = True
                should_stop = False

                if trainer.optimizer is not None:
                    current_lr = trainer.optimizer.param_groups[0]["lr"]
                    new_lr_from_server = current_lr * self._c6
                else:
                    new_lr_from_server = None

        if world_size > 1:
            device = trainer.model.device if hasattr(trainer, "model") else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            tensor = torch.zeros(4, dtype=torch.float32, device=device)

            if rank == 0:
                tensor[0] = 1.0 if should_stop else 0.0
                tensor[1] = lr_multiplier if lr_multiplier is not None else -1.0
                tensor[2] = new_lr_from_server if new_lr_from_server is not None else -1.0
                tensor[3] = 1.0 if use_fallback else 0.0

            dist.broadcast(tensor, src=0)

            if rank != 0:
                should_stop = bool(tensor[0].item())
                lr_multiplier_val = tensor[1].item()
                new_lr_val = tensor[2].item()
                use_fallback = bool(tensor[3].item())

                lr_multiplier = lr_multiplier_val if lr_multiplier_val >= 0 else None
                new_lr_from_server = new_lr_val if new_lr_val >= 0 else None

        if trainer.optimizer is not None:
            old_lr = trainer.optimizer.param_groups[0]["lr"]

            if new_lr_from_server is not None:
                final_lr = float(new_lr_from_server)

                for pg in trainer.optimizer.param_groups:
                    pg["lr"] = final_lr

        if should_stop:
            stop_summary = ""
            if control_to_apply is not None:
                stop_summary = control_to_apply.get("stop_summary", "")

            if rank == 0 and stop_summary:
                print(f"[EcoPhase] ✅ Stop summary:\n{stop_summary}")

            try:
                if hasattr(trainer, "control") and trainer.control is not None:
                    trainer.control.should_training_stop = True
                    trainer.control.should_epoch_stop = True

                if hasattr(trainer, "state") and trainer.state is not None:
                    trainer.state.is_training = False
                if hasattr(trainer, "_is_training"):
                    trainer._is_training = False
            except Exception:
                pass

    def _x0(self) -> None:
        for handle in self._a7:
            handle.remove()
        self._a7.clear()
