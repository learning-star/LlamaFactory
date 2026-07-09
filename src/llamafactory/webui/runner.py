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

import json
import os
import re
import sys
import time
from collections.abc import Generator
from copy import deepcopy
from subprocess import Popen
from threading import Lock
from typing import TYPE_CHECKING, Any

from transformers.utils import is_torch_npu_available

from ..extras.constants import (
    LLAMABOARD_CONFIG,
    MULTIMODAL_SUPPORTED_MODELS,
    PEFT_METHODS,
    RUNNING_LOG,
    SWANLAB_CONFIG,
    TRAINER_LOG,
    TRAINING_STAGES,
)
from ..extras.misc import is_accelerator_available, torch_gc
from ..extras.packages import is_gradio_available
from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONFIG_DIR,
    DEFAULT_SAVE_DIR,
    abort_process,
    calculate_pixels,
    gen_cmd,
    get_save_dir,
    get_time,
    load_args,
    load_config,
    load_eval_results,
    save_args,
    save_cmd,
)
from .control import get_compare_trainer_info, get_trainer_info
from .locales import ALERTS, LOCALES


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from .manager import Manager


class Runner:
    r"""A class to manage the running status of the trainers."""

    PLUGIN_STDOUT_LOG = f"{RUNNING_LOG}.stdout"

    def __init__(self, manager: "Manager", demo_mode: bool = False) -> None:
        r"""Init a runner."""
        self.manager = manager
        self.demo_mode = demo_mode
        """ Resume """
        self.trainers: dict[str, Popen] = {}
        self.run_output_paths: dict[str, str] = {}
        self.compare_mode = False
        self.do_train = True
        self.running_data: dict[Component, Any] = None
        """ State """
        self.aborted = False
        self.running = False
        self._state_lock = Lock()

    def set_abort(self) -> None:
        with self._state_lock:
            self.aborted = True
            trainers = list(self.trainers.values())

        for trainer in trainers:
            if trainer.poll() is None:
                abort_process(trainer.pid)

    def _begin_run(self, data: dict["Component", Any], do_train: bool) -> str:
        r"""Atomically validate and reserve the runner for one launch."""
        with self._state_lock:
            error = self._initialize(data, do_train, from_preview=False)
            if error:
                return error

            self.do_train, self.running_data = do_train, data
            self.aborted = False
            self.running = True
            return ""

    def _initialize(self, data: dict["Component", Any], do_train: bool, from_preview: bool) -> str:
        r"""Validate the configuration."""
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        lang, model_name, model_path = get("top.lang"), get("top.model_name"), get("top.model_path")
        dataset = get("train.dataset") if do_train else get("eval.dataset")

        if self.running:
            return ALERTS["err_conflict"][lang]

        if not model_name:
            return ALERTS["err_no_model"][lang]

        if not model_path:
            return ALERTS["err_no_path"][lang]

        if not dataset:
            return ALERTS["err_no_dataset"][lang]

        if not from_preview and self.demo_mode:
            return ALERTS["err_demo"][lang]

        if do_train:
            ecophase_username = str(get("train.ecophase_username") or "").strip()
            ecophase_api_key = str(get("train.ecophase_api_key") or "").strip()
            if not from_preview and (not ecophase_username or not ecophase_api_key):
                return ALERTS["err_no_ecophase_credentials"][lang]

            if not get("train.output_dir"):
                return ALERTS["err_no_output_dir"][lang]

            try:
                json.loads(get("train.extra_args"))
            except json.JSONDecodeError:
                return ALERTS["err_json_schema"][lang]

            stage = TRAINING_STAGES[get("train.training_stage")]
            if stage == "ppo" and not get("train.reward_model"):
                return ALERTS["err_no_reward_model"][lang]
        else:
            if not get("eval.output_dir"):
                return ALERTS["err_no_output_dir"][lang]

        if not from_preview and not is_accelerator_available():
            gr.Warning(ALERTS["warn_no_cuda"][lang])

        return ""

    def _finalize(self, lang: str, finish_info: str) -> None:
        r"""Clean the cached memory and resets the runner."""
        finish_info = ALERTS["info_aborted"][lang] if self.aborted else finish_info
        gr.Info(finish_info)
        with self._state_lock:
            self.trainers = {}
            self.run_output_paths = {}
            self.compare_mode = False
            self.aborted = False
            self.running = False
            self.running_data = None
        torch_gc()

    def _launch_trainer(
        self,
        args: dict[str, Any],
        output_dir: str,
        enable_plugin: bool = False,
        plugin_username: str | None = None,
        plugin_api_key: str | None = None,
        cuda_visible_devices: str | None = None,
    ) -> Popen:
        r"""Launch one trainer subprocess."""
        env = deepcopy(os.environ)
        env["LLAMABOARD_ENABLED"] = "1"
        env["LLAMABOARD_WORKDIR"] = output_dir
        if cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        if enable_plugin:
            env["ECOPHASE_AI_PLUGIN"] = "1"
            if plugin_username:
                env["ECO_CLIENT_ID"] = plugin_username
                env["ECOPHASE_AI_USERNAME"] = plugin_username
            else:
                env.pop("ECO_CLIENT_ID", None)
                env.pop("ECOPHASE_AI_USERNAME", None)
            if plugin_api_key:
                env["ECO_API_KEY"] = plugin_api_key
                env["ECOPHASE_AI_API_KEY"] = plugin_api_key
            else:
                env.pop("ECO_API_KEY", None)
                env.pop("ECOPHASE_AI_API_KEY", None)
        else:
            env.pop("ECO_CLIENT_ID", None)
            env.pop("ECO_API_KEY", None)
            env.pop("ECO_GRPC_ADDR", None)
            env.pop("ECO_TLS_ROOT_CA", None)
            env.pop("ECOPHASE_AI_PLUGIN", None)
            env.pop("ECOPHASE_AI_USERNAME", None)
            env.pop("ECOPHASE_AI_API_KEY", None)

        if args.get("deepspeed", None) is not None:
            env["FORCE_TORCHRUN"] = "1"

        os.makedirs(output_dir, exist_ok=True)
        stdout_log = open(os.path.join(output_dir, self.PLUGIN_STDOUT_LOG), "a", encoding="utf-8", buffering=1)
        try:
            return Popen(
                [sys.executable, "-m", "llamafactory.cli", "train", save_cmd(args)],
                env=env,
                stdout=stdout_log,
                stderr=stdout_log,
                text=True,
            )
        finally:
            stdout_log.close()

    def _read_running_log(self, output_dir: str) -> str:
        r"""Read the latest running log text for one trainer."""
        running_logs = []
        for file_name in (RUNNING_LOG, self.PLUGIN_STDOUT_LOG):
            running_log_path = os.path.join(output_dir, file_name)
            if not os.path.isfile(running_log_path):
                continue

            with open(running_log_path, encoding="utf-8") as f:
                running_logs.append(f.read()[-20000:])

        return "\n".join(running_logs)[-30000:]

    def _check_plugin_log_status(self, output_dir: str) -> str | None:
        r"""Return plugin startup status from running logs."""
        running_log = self._read_running_log(output_dir)
        if "API is disabled" in running_log:
            return "disabled"

        if "API is enabled" in running_log:
            return "enabled"

        return None

    def _wait_for_plugin_startup(self, trainer: Popen, output_dir: str) -> str:
        r"""Wait until the plugin trainer confirms API enablement or exits."""
        timeout = int(os.getenv("ECOPHASE_PLUGIN_STARTUP_TIMEOUT_SECONDS", "120"))
        deadline = time.time() + timeout
        while time.time() < deadline:
            return_code = trainer.poll()
            if return_code is not None:
                _, stderr = trainer.communicate()
                stderr = stderr or ""
                return f"EcoTrain Plugin failed to start. Exit code: {return_code}\n\n```\n{stderr}\n```"

            if self.aborted:
                return "EcoTrain Plugin startup was aborted."

            plugin_status = self._check_plugin_log_status(output_dir)
            if plugin_status == "enabled":
                return ""
            if plugin_status == "disabled":
                abort_process(trainer.pid)
                _, stderr = trainer.communicate()
                stderr = stderr or ""
                running_log = self._read_running_log(output_dir)
                return (
                    "EcoTrain Plugin reported API disabled. Baseline was not started.\n\n"
                    f"```\n{running_log}\n{stderr}\n```"
                )

            time.sleep(2)

        abort_process(trainer.pid)
        _, stderr = trainer.communicate()
        stderr = stderr or ""
        return (
            "EcoTrain Plugin startup timed out before API enabled status was confirmed. "
            "Baseline was not started.\n\n"
            f"```\n{stderr}\n```"
        )

    def _apply_user_output_root(self, args: dict[str, Any], plugin_username: str | None) -> None:
        r"""Move EcoPhase training outputs under a user-scoped root when configured."""
        output_root = os.getenv("ECOPHASE_OUTPUT_ROOT", "").strip()
        if not output_root or not plugin_username:
            return

        user_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", plugin_username).strip("._-") or "unknown"
        output_dir = os.path.normpath(str(args["output_dir"]))
        save_root = os.path.normpath(DEFAULT_SAVE_DIR)
        if os.path.isabs(output_dir):
            suffix = os.path.basename(output_dir)
        else:
            rel_output = os.path.relpath(output_dir, save_root)
            suffix = os.path.basename(output_dir) if rel_output.startswith("..") else rel_output

        args["output_dir"] = os.path.join(output_root, user_slug, suffix)

    def _build_training_runs(
        self, args: dict[str, Any], output_path: str, plugin_enabled: bool
    ) -> dict[str, tuple[dict[str, Any], str, bool]]:
        r"""Build trainer arguments and target directories for each run."""
        if not plugin_enabled:
            return {
                "Baseline": (
                    {**args, "output_dir": output_path},
                    output_path,
                    False,
                )
            }

        baseline_dir = os.path.join(output_path, "baseline")
        plugin_dir = os.path.join(output_path, "ecophase")
        return {
            "Baseline": (
                {**args, "output_dir": baseline_dir},
                baseline_dir,
                False,
            ),
            "EcoTrain Plugin": (
                {**args, "output_dir": plugin_dir},
                plugin_dir,
                True,
            ),
        }

    def _build_run_output_path(self, output_path: str) -> str:
        r"""Create a unique run directory under the user-selected output path."""
        run_slug = f"run_{get_time()}_{int(time.time() * 1000) % 1000:03d}_{os.getpid()}"
        candidate = os.path.join(output_path, run_slug)
        suffix = 1
        while os.path.exists(candidate):
            suffix += 1
            candidate = os.path.join(output_path, f"{run_slug}_{suffix}")

        return candidate

    def _parse_train_args(self, data: dict["Component", Any]) -> dict[str, Any]:
        r"""Build and validate the training arguments."""
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        model_name, finetuning_type = get("top.model_name"), get("top.finetuning_type")
        user_config = load_config()

        args = dict(
            stage=TRAINING_STAGES[get("train.training_stage")],
            do_train=True,
            model_name_or_path=get("top.model_path"),
            cache_dir=user_config.get("cache_dir", None),
            preprocessing_num_workers=16,
            finetuning_type=finetuning_type,
            template=get("top.template"),
            rope_scaling=get("top.rope_scaling") if get("top.rope_scaling") != "none" else None,
            flash_attn="fa2" if get("top.booster") == "flashattn2" else "auto",
            use_unsloth=(get("top.booster") == "unsloth"),
            enable_liger_kernel=(get("top.booster") == "liger_kernel"),
            dataset_dir=get("train.dataset_dir"),
            dataset=",".join(get("train.dataset")),
            cutoff_len=get("train.cutoff_len"),
            learning_rate=float(get("train.learning_rate")),
            num_train_epochs=float(get("train.num_train_epochs")),
            max_samples=int(get("train.max_samples")),
            per_device_train_batch_size=get("train.batch_size"),
            gradient_accumulation_steps=get("train.gradient_accumulation_steps"),
            lr_scheduler_type=get("train.lr_scheduler_type"),
            max_grad_norm=float(get("train.max_grad_norm")),
            logging_steps=get("train.logging_steps"),
            save_steps=get("train.save_steps"),
            warmup_steps=get("train.warmup_steps"),
            neftune_noise_alpha=get("train.neftune_alpha") or None,
            packing=get("train.packing") or get("train.neat_packing"),
            neat_packing=get("train.neat_packing"),
            train_on_prompt=get("train.train_on_prompt"),
            mask_history=get("train.mask_history"),
            resize_vocab=get("train.resize_vocab"),
            use_llama_pro=get("train.use_llama_pro"),
            enable_thinking=get("train.enable_thinking"),
            report_to=get("train.report_to"),
            use_galore=get("train.use_galore"),
            use_apollo=get("train.use_apollo"),
            use_badam=get("train.use_badam"),
            use_swanlab=get("train.use_swanlab"),
            output_dir=get_save_dir(model_name, finetuning_type, get("train.output_dir")),
            fp16=(get("train.compute_type") == "fp16"),
            bf16=(get("train.compute_type") == "bf16"),
            pure_bf16=(get("train.compute_type") == "pure_bf16"),
            plot_loss=True,
            trust_remote_code=True,
            ddp_timeout=180000000,
            include_num_input_tokens_seen=True,
        )
        args.update(json.loads(get("train.extra_args")))
        args["save_only_model"] = True

        # checkpoints
        if get("top.checkpoint_path"):
            if finetuning_type in PEFT_METHODS:  # list
                args["adapter_name_or_path"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in get("top.checkpoint_path")]
                )
            else:  # str
                args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, get("top.checkpoint_path"))

        # quantization
        if get("top.quantization_bit") != "none":
            args["quantization_bit"] = int(get("top.quantization_bit"))
            args["quantization_method"] = get("top.quantization_method")
            args["double_quantization"] = not is_torch_npu_available()

        # freeze config
        if args["finetuning_type"] == "freeze":
            args["freeze_trainable_layers"] = get("train.freeze_trainable_layers")
            args["freeze_trainable_modules"] = get("train.freeze_trainable_modules")
            args["freeze_extra_modules"] = get("train.freeze_extra_modules") or None

        # lora config
        if args["finetuning_type"] == "lora":
            args["lora_rank"] = get("train.lora_rank")
            args["lora_alpha"] = get("train.lora_alpha")
            args["lora_dropout"] = get("train.lora_dropout")
            args["loraplus_lr_ratio"] = get("train.loraplus_lr_ratio") or None
            args["create_new_adapter"] = get("train.create_new_adapter")
            args["use_rslora"] = get("train.use_rslora")
            args["use_dora"] = get("train.use_dora")
            args["pissa_init"] = get("train.use_pissa")
            args["pissa_convert"] = get("train.use_pissa")
            args["lora_target"] = get("train.lora_target") or "all"
            args["additional_target"] = get("train.additional_target") or None

            if args["use_llama_pro"]:
                args["freeze_trainable_layers"] = get("train.freeze_trainable_layers")

        # rlhf config
        if args["stage"] == "ppo":
            if finetuning_type in PEFT_METHODS:
                args["reward_model"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in get("train.reward_model")]
                )
            else:
                args["reward_model"] = get_save_dir(model_name, finetuning_type, get("train.reward_model"))

            args["reward_model_type"] = "lora" if finetuning_type == "lora" else "full"
            args["ppo_score_norm"] = get("train.ppo_score_norm")
            args["ppo_whiten_rewards"] = get("train.ppo_whiten_rewards")
            args["top_k"] = 0
            args["top_p"] = 0.9
        elif args["stage"] in ["dpo", "kto"]:
            args["pref_beta"] = get("train.pref_beta")
            args["pref_ftx"] = get("train.pref_ftx")
            args["pref_loss"] = get("train.pref_loss")

        # multimodal config
        if model_name in MULTIMODAL_SUPPORTED_MODELS:
            args["freeze_vision_tower"] = get("train.freeze_vision_tower")
            args["freeze_multi_modal_projector"] = get("train.freeze_multi_modal_projector")
            args["freeze_language_model"] = get("train.freeze_language_model")
            args["image_max_pixels"] = calculate_pixels(get("train.image_max_pixels"))
            args["image_min_pixels"] = calculate_pixels(get("train.image_min_pixels"))
            args["video_max_pixels"] = calculate_pixels(get("train.video_max_pixels"))
            args["video_min_pixels"] = calculate_pixels(get("train.video_min_pixels"))

        # galore config
        if args["use_galore"]:
            args["galore_rank"] = get("train.galore_rank")
            args["galore_update_interval"] = get("train.galore_update_interval")
            args["galore_scale"] = get("train.galore_scale")
            args["galore_target"] = get("train.galore_target")

        # apollo config
        if args["use_apollo"]:
            args["apollo_rank"] = get("train.apollo_rank")
            args["apollo_update_interval"] = get("train.apollo_update_interval")
            args["apollo_scale"] = get("train.apollo_scale")
            args["apollo_target"] = get("train.apollo_target")

        # badam config
        if args["use_badam"]:
            args["badam_mode"] = get("train.badam_mode")
            args["badam_switch_mode"] = get("train.badam_switch_mode")
            args["badam_switch_interval"] = get("train.badam_switch_interval")
            args["badam_update_ratio"] = get("train.badam_update_ratio")

        # swanlab config
        if get("train.use_swanlab"):
            args["swanlab_project"] = get("train.swanlab_project")
            args["swanlab_run_name"] = get("train.swanlab_run_name")
            args["swanlab_workspace"] = get("train.swanlab_workspace")
            args["swanlab_api_key"] = get("train.swanlab_api_key")
            args["swanlab_mode"] = get("train.swanlab_mode")

        # eval config
        if args["stage"] != "ppo":
            val_size = float(get("train.val_size"))
            args["val_size"] = val_size
            if val_size > 0:
                args["eval_strategy"] = "steps"
                args["eval_steps"] = int(os.getenv("ECOPHASE_EVAL_STEPS", "20"))
                args["per_device_eval_batch_size"] = int(os.getenv("ECOPHASE_EVAL_BATCH_SIZE", "16"))
                args["compute_accuracy"] = True
                args["eval_on_start"] = True

        # ds config
        if get("train.ds_stage") != "none":
            ds_stage = get("train.ds_stage")
            ds_offload = "offload_" if get("train.ds_offload") else ""
            args["deepspeed"] = os.path.join(DEFAULT_CACHE_DIR, f"ds_z{ds_stage}_{ds_offload}config.json")

        return args

    def _parse_eval_args(self, data: dict["Component", Any]) -> dict[str, Any]:
        r"""Build and validate the evaluation arguments."""
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        model_name, finetuning_type = get("top.model_name"), get("top.finetuning_type")
        user_config = load_config()

        args = dict(
            stage="sft",
            model_name_or_path=get("top.model_path"),
            cache_dir=user_config.get("cache_dir", None),
            preprocessing_num_workers=16,
            finetuning_type=finetuning_type,
            quantization_method=get("top.quantization_method"),
            template=get("top.template"),
            rope_scaling=get("top.rope_scaling") if get("top.rope_scaling") != "none" else None,
            flash_attn="fa2" if get("top.booster") == "flashattn2" else "auto",
            use_unsloth=(get("top.booster") == "unsloth"),
            dataset_dir=get("eval.dataset_dir"),
            eval_dataset=",".join(get("eval.dataset")),
            cutoff_len=get("eval.cutoff_len"),
            max_samples=int(get("eval.max_samples")),
            per_device_eval_batch_size=get("eval.batch_size"),
            predict_with_generate=True,
            report_to="none",
            max_new_tokens=get("eval.max_new_tokens"),
            top_p=get("eval.top_p"),
            temperature=get("eval.temperature"),
            output_dir=get_save_dir(model_name, finetuning_type, get("eval.output_dir")),
            trust_remote_code=True,
            ddp_timeout=180000000,
        )

        if get("eval.predict"):
            args["do_predict"] = True
        else:
            args["do_eval"] = True

        # checkpoints
        if get("top.checkpoint_path"):
            if finetuning_type in PEFT_METHODS:  # list
                args["adapter_name_or_path"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in get("top.checkpoint_path")]
                )
            else:  # str
                args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, get("top.checkpoint_path"))

        # quantization
        if get("top.quantization_bit") != "none":
            args["quantization_bit"] = int(get("top.quantization_bit"))
            args["quantization_method"] = get("top.quantization_method")
            args["double_quantization"] = not is_torch_npu_available()

        return args

    def _preview(self, data: dict["Component", Any], do_train: bool) -> Generator[dict["Component", str], None, None]:
        r"""Preview the training commands."""
        output_box = self.manager.get_elem_by_id("{}.output_box".format("train" if do_train else "eval"))
        error = self._initialize(data, do_train, from_preview=True)
        if error:
            gr.Warning(error)
            output_dict = {output_box: error}
        else:
            args = self._parse_train_args(data) if do_train else self._parse_eval_args(data)
            output_dict = {output_box: gen_cmd(args)}

        if do_train:
            self._show_train_output(output_dict)

        yield output_dict

    def _show_train_output(self, output_dict: dict["Component", Any]) -> dict["Component", Any]:
        r"""Show the single training output box and hide compare output."""
        output_row_single = self.manager.get_elem_by_id("train.output_row_single")
        output_row_compare = self.manager.get_elem_by_id("train.output_row_compare")
        output_dict[output_row_single] = gr.update(visible=True)
        output_dict[output_row_compare] = gr.update(visible=False)
        return output_dict

    def _clear_train_plots(self, output_dict: dict["Component", Any]) -> dict["Component", Any]:
        r"""Clear stale training metric plots before a new run starts."""
        for elem_id in ("train.loss_viewer", "train.eval_loss_viewer", "train.eval_accuracy_viewer"):
            output_dict[self.manager.get_elem_by_id(elem_id)] = gr.update(value=None)

        return output_dict

    def _clear_training_monitor_artifacts(self, output_dirs: list[str]) -> None:
        r"""Remove stale monitor artifacts so a restarted run redraws plots from fresh logs."""
        for output_dir in output_dirs:
            for file_name in (RUNNING_LOG, self.PLUGIN_STDOUT_LOG, TRAINER_LOG, SWANLAB_CONFIG):
                try:
                    os.remove(os.path.join(output_dir, file_name))
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    def _set_train_button_state(self, output_dict: dict["Component", Any], running: bool) -> dict["Component", Any]:
        r"""Update train action buttons for the current run state."""
        output_dict[self.manager.get_elem_by_id("train.start_btn")] = gr.update(interactive=not running)
        output_dict[self.manager.get_elem_by_id("train.stop_btn")] = gr.update(interactive=running)
        return output_dict

    def _launch(self, data: dict["Component", Any], do_train: bool) -> Generator[dict["Component", Any], None, None]:
        r"""Start the training process."""
        output_box = self.manager.get_elem_by_id("{}.output_box".format("train" if do_train else "eval"))
        error = self._begin_run(data, do_train)
        if error:
            gr.Warning(error)
            yield {output_box: error}
        else:
            args = self._parse_train_args(data) if do_train else self._parse_eval_args(data)
            plugin_enabled = do_train
            plugin_username = data[self.manager.get_elem_by_id("train.ecophase_username")] if do_train else None
            plugin_api_key = data[self.manager.get_elem_by_id("train.ecophase_api_key")] if do_train else None

            if do_train:
                yield self._clear_train_plots(
                    self._set_train_button_state(
                        {
                            output_box: "",
                            self.manager.get_elem_by_id("train.progress_bar"): gr.Slider(visible=False),
                            self.manager.get_elem_by_id("train.output_row_single"): gr.update(visible=True),
                            self.manager.get_elem_by_id("train.output_row_compare"): gr.update(visible=False),
                        },
                        running=True,
                    )
                )
                self._apply_user_output_root(args, plugin_username)
                gr.Info(ALERTS["info_ecophase_dual_start"][data[self.manager.get_elem_by_id("top.lang")]])

            output_path = self._build_run_output_path(args["output_dir"]) if do_train else args["output_dir"]
            os.makedirs(args["output_dir"], exist_ok=True)
            os.makedirs(output_path, exist_ok=True)
            config_dict = self._build_config_dict(data)
            save_args(os.path.join(args["output_dir"], LLAMABOARD_CONFIG), config_dict)
            if output_path != args["output_dir"]:
                save_args(os.path.join(output_path, LLAMABOARD_CONFIG), config_dict)

            self.compare_mode = plugin_enabled
            self.trainers = {}
            self.run_output_paths = {}

            run_specs = self._build_training_runs(args, output_path, plugin_enabled)
            for label, (_, run_output_dir, _) in run_specs.items():
                os.makedirs(run_output_dir, exist_ok=True)
                self.run_output_paths[label] = run_output_dir

            if do_train:
                monitor_dirs = list(dict.fromkeys([args["output_dir"], output_path, *self.run_output_paths.values()]))
                self._clear_training_monitor_artifacts(monitor_dirs)

            if plugin_enabled:
                plugin_label = "EcoTrain Plugin"
                plugin_args, plugin_output_dir, _ = run_specs[plugin_label]
                self.trainers[plugin_label] = self._launch_trainer(
                    plugin_args,
                    plugin_output_dir,
                    enable_plugin=True,
                    plugin_username=plugin_username,
                    plugin_api_key=plugin_api_key,
                    cuda_visible_devices=os.getenv("ECOPHASE_PLUGIN_CUDA_VISIBLE_DEVICES", "1"),
                )
                startup_error = self._wait_for_plugin_startup(self.trainers[plugin_label], plugin_output_dir)
                if startup_error:
                    lang = data[self.manager.get_elem_by_id("top.lang")]
                    startup_output = {output_box: startup_error}
                    running_log, running_progress, running_info = get_trainer_info(lang, plugin_output_dir, True)
                    if running_log:
                        startup_output[output_box] = startup_error + "\n\n" + running_log

                    startup_output[self.manager.get_elem_by_id("train.progress_bar")] = running_progress
                    for elem_id, info_key in (
                        ("train.loss_viewer", "loss_viewer"),
                        ("train.eval_loss_viewer", "eval_loss_viewer"),
                        ("train.eval_accuracy_viewer", "eval_accuracy_viewer"),
                        ("train.swanlab_link", "swanlab_link"),
                    ):
                        if info_key in running_info:
                            startup_output[self.manager.get_elem_by_id(elem_id)] = running_info[info_key]

                    self._finalize(lang, ALERTS["err_failed"][lang])
                    yield self._set_train_button_state(startup_output, running=False)
                    return

                baseline_label = "Baseline"
                baseline_args, baseline_output_dir, _ = run_specs[baseline_label]
                # NOTE: DO NOT USE shell=True to avoid security risk
                self.trainers[baseline_label] = self._launch_trainer(
                    baseline_args,
                    baseline_output_dir,
                    enable_plugin=False,
                    cuda_visible_devices=os.getenv("ECOPHASE_BASELINE_CUDA_VISIBLE_DEVICES", "0"),
                )
            else:
                for label, (run_args, run_output_dir, enable_plugin) in run_specs.items():
                    # NOTE: DO NOT USE shell=True to avoid security risk
                    self.trainers[label] = self._launch_trainer(
                        run_args,
                        run_output_dir,
                        enable_plugin=enable_plugin,
                    )

            yield from self.monitor()

    def _build_config_dict(self, data: dict["Component", Any]) -> dict[str, Any]:
        r"""Build a dictionary containing the current training configuration."""
        config_dict = {}
        skip_ids = ["top.lang", "top.model_path", "train.output_dir", "train.config_path", "train.ecophase_api_key"]
        for elem, value in data.items():
            elem_id = self.manager.get_id_by_elem(elem)
            if elem_id not in skip_ids:
                config_dict[elem_id] = value

        return config_dict

    def preview_train(self, data):
        yield from self._preview(data, do_train=True)

    def preview_eval(self, data):
        yield from self._preview(data, do_train=False)

    def run_train(self, data):
        yield from self._launch(data, do_train=True)

    def run_eval(self, data):
        yield from self._launch(data, do_train=False)

    def monitor(self):
        r"""Monitorgit the training progress and logs."""
        with self._state_lock:
            self.running = True

        get = lambda elem_id: self.running_data[self.manager.get_elem_by_id(elem_id)]
        lang, model_name, finetuning_type = get("top.lang"), get("top.model_name"), get("top.finetuning_type")
        output_dir = get("{}.output_dir".format("train" if self.do_train else "eval"))
        output_path = get_save_dir(model_name, finetuning_type, output_dir)

        output_box = self.manager.get_elem_by_id("{}.output_box".format("train" if self.do_train else "eval"))
        output_row_single = self.manager.get_elem_by_id("train.output_row_single") if self.do_train else None
        output_row_compare = self.manager.get_elem_by_id("train.output_row_compare") if self.do_train else None
        output_box_compare_left = self.manager.get_elem_by_id("train.output_box_compare_left") if self.do_train else None
        output_box_compare_right = self.manager.get_elem_by_id("train.output_box_compare_right") if self.do_train else None
        progress_bar = self.manager.get_elem_by_id("{}.progress_bar".format("train" if self.do_train else "eval"))
        loss_viewer = self.manager.get_elem_by_id("train.loss_viewer") if self.do_train else None
        eval_loss_viewer = self.manager.get_elem_by_id("train.eval_loss_viewer") if self.do_train else None
        eval_accuracy_viewer = self.manager.get_elem_by_id("train.eval_accuracy_viewer") if self.do_train else None
        swanlab_link = self.manager.get_elem_by_id("train.swanlab_link") if self.do_train else None

        running_log = ""
        running_logs: dict[str, str] = {}
        return_codes: dict[str, int] = {}
        while True:
            if self.aborted:
                abort_dict = {
                    output_box: ALERTS["info_aborting"][lang],
                    progress_bar: gr.Slider(visible=False),
                }
                if output_row_single is not None and output_row_compare is not None:
                    abort_dict[output_row_single] = gr.update(visible=True)
                    abort_dict[output_row_compare] = gr.update(visible=False)
                if output_box_compare_left is not None and output_box_compare_right is not None:
                    abort_dict[output_box_compare_left] = gr.update(value="")
                    abort_dict[output_box_compare_right] = gr.update(value="")
                if self.do_train:
                    self._set_train_button_state(abort_dict, running=True)
                yield self._clear_train_plots(abort_dict) if self.do_train else abort_dict
            else:
                if self.compare_mode:
                    running_logs, running_progress, running_info = get_compare_trainer_info(
                        lang, self.run_output_paths, self.do_train
                    )
                else:
                    primary_output_path = next(iter(self.run_output_paths.values()), output_path)
                    running_log, running_progress, running_info = get_trainer_info(
                        lang, primary_output_path, self.do_train
                    )
                return_dict = {
                    output_box: running_log,
                    progress_bar: running_progress,
                }
                if output_row_single is not None and output_row_compare is not None:
                    if self.compare_mode and output_box_compare_left is not None and output_box_compare_right is not None:
                        compare_labels = list(self.run_output_paths.keys())
                        left_log = running_logs.get(compare_labels[0], "") if len(compare_labels) > 0 else ""
                        right_log = running_logs.get(compare_labels[1], "") if len(compare_labels) > 1 else ""
                        return_dict[output_row_single] = gr.update(visible=False)
                        return_dict[output_row_compare] = gr.update(visible=True)
                        return_dict[output_box_compare_left] = left_log
                        return_dict[output_box_compare_right] = right_log
                    else:
                        return_dict[output_row_single] = gr.update(visible=True)
                        return_dict[output_row_compare] = gr.update(visible=False)
                if "loss_viewer" in running_info:
                    return_dict[loss_viewer] = running_info["loss_viewer"]

                if "eval_loss_viewer" in running_info:
                    return_dict[eval_loss_viewer] = running_info["eval_loss_viewer"]

                if "eval_accuracy_viewer" in running_info:
                    return_dict[eval_accuracy_viewer] = running_info["eval_accuracy_viewer"]

                if "swanlab_link" in running_info:
                    return_dict[swanlab_link] = running_info["swanlab_link"]

                yield return_dict

            all_finished = True
            for label, trainer in self.trainers.items():
                return_code = trainer.poll()
                if return_code is None:
                    all_finished = False
                else:
                    return_codes[label] = return_code

            if any(return_code not in (None, 0) for return_code in return_codes.values()) and not self.aborted:
                for trainer in self.trainers.values():
                    if trainer.poll() is None:
                        abort_process(trainer.pid)
                all_finished = False

            if all_finished:
                break

            time.sleep(2)

        stderrs: dict[str, str] = {}
        for label, trainer in self.trainers.items():
            _, stderr = trainer.communicate()
            stderrs[label] = stderr or ""
            if label not in return_codes:
                return_codes[label] = trainer.returncode or 0

        was_aborted = self.aborted
        if was_aborted:
            finish_info = ALERTS["info_aborted"][lang]
            finish_log = ALERTS["info_aborted"][lang] + "\n\n" + running_log
        elif all(return_code == 0 for return_code in return_codes.values()):
            finish_info = ALERTS["info_finished"][lang]
            if self.do_train:
                finish_log = ALERTS["info_finished"][lang] + "\n\n" + running_log
            else:
                finish_log = load_eval_results(os.path.join(output_path, "all_results.json")) + "\n\n" + running_log
        else:
            failed_label = next((label for label, return_code in return_codes.items() if return_code != 0), "unknown")
            failed_stderr = stderrs.get(failed_label, "")
            print(failed_stderr)
            finish_info = ALERTS["err_failed"][lang]
            finish_log = (
                ALERTS["err_failed"][lang]
                + f" [{failed_label}] Exit code: {return_codes.get(failed_label, -1)}\n\n```\n{failed_stderr}\n```"
            )

        self._finalize(lang, finish_info)
        return_dict = {output_box: finish_log, progress_bar: gr.Slider(visible=False)}
        if output_row_single is not None and output_row_compare is not None:
            if self.compare_mode and output_box_compare_left is not None and output_box_compare_right is not None:
                compare_labels = list(self.run_output_paths.keys())
                left_log = running_logs.get(compare_labels[0], "") if len(compare_labels) > 0 else ""
                right_log = running_logs.get(compare_labels[1], "") if len(compare_labels) > 1 else ""
                return_dict[output_row_single] = gr.update(visible=False)
                return_dict[output_row_compare] = gr.update(visible=True)
                return_dict[output_box_compare_left] = left_log
                return_dict[output_box_compare_right] = right_log
            else:
                return_dict[output_row_single] = gr.update(visible=True)
                return_dict[output_row_compare] = gr.update(visible=False)
        if was_aborted and self.do_train:
            self._clear_train_plots(return_dict)
        if self.do_train:
            self._set_train_button_state(return_dict, running=False)
        yield return_dict

    def save_args(self, data):
        r"""Save the training configuration to config path."""
        output_box = self.manager.get_elem_by_id("train.output_box")
        error = self._initialize(data, do_train=True, from_preview=True)
        if error:
            gr.Warning(error)
            return self._show_train_output({output_box: error})

        lang = data[self.manager.get_elem_by_id("top.lang")]
        config_path_elem = self.manager.get_elem_by_id("train.config_path")
        config_path = data[config_path_elem] or f"{data[self.manager.get_elem_by_id('train.current_time')]}.yaml"
        os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
        save_path = os.path.join(DEFAULT_CONFIG_DIR, config_path)

        save_args(save_path, self._build_config_dict(data))
        return self._show_train_output({output_box: ALERTS["info_config_saved"][lang] + save_path, config_path_elem: config_path})

    def load_args(self, lang: str, config_path: str):
        r"""Load the training configuration from config path."""
        output_box = self.manager.get_elem_by_id("train.output_box")
        config_dict = load_args(os.path.join(DEFAULT_CONFIG_DIR, config_path))
        if config_dict is None:
            gr.Warning(ALERTS["err_config_not_found"][lang])
            return self._show_train_output({output_box: ALERTS["err_config_not_found"][lang]})

        output_dict: dict[Component, Any] = {output_box: ALERTS["info_config_loaded"][lang]}
        for elem_id, value in config_dict.items():
            try:
                output_dict[self.manager.get_elem_by_id(elem_id)] = value
            except KeyError:
                continue

        return self._show_train_output(output_dict)

    def check_output_dir(self, lang: str, model_name: str, finetuning_type: str, output_dir: str):
        r"""Restore the training status if output_dir exists."""
        output_box = self.manager.get_elem_by_id("train.output_box")
        output_dict: dict[Component, Any] = {output_box: LOCALES["output_box"][lang]["value"]}
        if model_name and output_dir and os.path.isdir(get_save_dir(model_name, finetuning_type, output_dir)):
            gr.Warning(ALERTS["warn_output_dir_exists"][lang])
            output_dict[output_box] = ALERTS["warn_output_dir_exists"][lang]

            output_dir = get_save_dir(model_name, finetuning_type, output_dir)
            config_dict = load_args(os.path.join(output_dir, LLAMABOARD_CONFIG))  # load llamaboard config
            for elem_id, value in config_dict.items():
                output_dict[self.manager.get_elem_by_id(elem_id)] = value

        return output_dict
