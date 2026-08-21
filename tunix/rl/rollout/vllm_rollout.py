# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM rollout worker with Tunix sampler."""

from absl import logging
import hashlib
import os
import pickle
import socket
import struct
import threading
import time
from typing import Any, Dict, Mapping, Optional, Tuple

from flax import nnx
import jax
import jaxtyping
import numpy as np
from tunix.generate import mappings
from tunix.generate import vllm_sampler
from tunix.rl.rollout import base_rollout


def _send_message(sock: socket.socket, payload: Any) -> None:
  message = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
  sock.sendall(struct.pack("!Q", len(message)))
  sock.sendall(message)


def _recv_exactly(sock: socket.socket, num_bytes: int) -> bytes:
  chunks = []
  remaining = num_bytes
  while remaining > 0:
    chunk = sock.recv(remaining)
    if not chunk:
      raise ConnectionError("Socket connection closed before payload arrived.")
    chunks.append(chunk)
    remaining -= len(chunk)
  return b"".join(chunks)


def _recv_message(sock: socket.socket) -> Any:
  payload_size = struct.unpack("!Q", _recv_exactly(sock, 8))[0]
  return pickle.loads(_recv_exactly(sock, payload_size))


def _host_transport_port() -> int:
  return int(os.environ.get("TUNIX_DISTRIBUTED_ROLLOUT_PORT", "29600"))


def _serialize_rollout_output(
    output: base_rollout.RolloutOutput,
) -> dict[str, Any]:
  return {
      "text": output.text,
      "logits": None
      if output.logits is None
      else [np.asarray(x) for x in output.logits],
      "tokens": [np.asarray(x) for x in output.tokens],
      "left_padded_prompt_tokens": np.asarray(output.left_padded_prompt_tokens),
      "logprobs": None
      if output.logprobs is None
      else [np.asarray(x) for x in output.logprobs],
  }


def _deserialize_rollout_output(
    payload: dict[str, Any],
) -> base_rollout.RolloutOutput:
  return base_rollout.RolloutOutput(
      text=list(payload["text"]),
      logits=payload["logits"],
      tokens=list(payload["tokens"]),
      left_padded_prompt_tokens=np.asarray(
          payload["left_padded_prompt_tokens"]
      ),
      logprobs=payload["logprobs"],
  )


def _serialize_state_for_transport(state: nnx.State) -> list[tuple[Any, Any]]:
  flat_state = []
  for path, value in state.flat_state():
    if hasattr(value, "value"):
      value = value.value
    if isinstance(value, jax.Array):
      if not value.is_fully_addressable:
        raise ValueError(
            "Distributed rollout transport requires fully addressable source"
            " state on the sending process."
        )
      value = np.asarray(value)
    flat_state.append((tuple(path), value))
  return flat_state


def _deserialize_state_from_transport(
    payload: list[tuple[Any, Any]],
) -> nnx.State:
  return nnx.State.from_flat_path(
      (path, nnx.Param(value)) for path, value in payload
  )


def _stable_request_id(
    internal_request_tags: Mapping[str, Any] | None,
    request_counter: int,
) -> str:
  if not internal_request_tags:
    return f"request-{request_counter}"
  digest = hashlib.sha256(
      repr(sorted(internal_request_tags.items())).encode("utf-8")
  ).hexdigest()
  return digest[:32]


class VllmRollout(base_rollout.BaseRollout):
  """vLLM rollout worker."""

  def __init__(
      self,
      model: Any,
      tokenizer: Any,
      cache_config_or_size: base_rollout.CacheConfig | int,
      mesh: jax.sharding.Mesh,
      rollout_config: base_rollout.RolloutConfig,
      *,
      load_initial_checkpoint: bool = True,
  ):
    mapping_config = mappings.MappingConfig.build(
        mapping_obj=rollout_config.rollout_mapping_config,
        model=model,
        backend="vllm_jax",
    )
    self._sampler = vllm_sampler.VllmSampler(
        tokenizer=tokenizer,
        config=vllm_sampler.VllmConfig(
            server_mode=rollout_config.rollout_vllm_server_mode,
            mapping_config=mapping_config,
            return_logprobs=rollout_config.return_logprobs,
            init_with_random_weights=rollout_config.rollout_vllm_init_with_random_weights,
            tpu_backend_type=rollout_config.rollout_vllm_tpu_backend_type,
            additional_config=rollout_config.rollout_vllm_additional_config,
            enable_dp_attention=rollout_config.rollout_vllm_enable_dp_attention,
            hbm_utilization=rollout_config.rollout_vllm_hbm_utilization,
            lora_config=rollout_config.rollout_vllm_lora_config,
            mesh=mesh,
            tensor_parallel_size=rollout_config.tensor_parallel_size,
            data_parallel_size=rollout_config.data_parallel_size,
            expert_parallel_size=rollout_config.expert_parallel_size,
            engine_kwargs={
                "model": rollout_config.rollout_vllm_model_version,
                "max_model_len": cache_config_or_size,
                "async_scheduling": (
                    rollout_config.rollout_vllm_async_scheduling
                ),
                "max_num_batched_tokens": (
                    rollout_config.rollout_vllm_max_num_batched_tokens
                ),
                "max_num_seqs": rollout_config.rollout_vllm_max_num_seqs,
                "hf_config_path": rollout_config.rollout_vllm_hf_config_path,
                "max_logprobs": (
                    1
                ),  # We only need the logprobs of the sampled tokens
                **rollout_config.rollout_vllm_kwargs,
            },
            sampling_kwargs=rollout_config.rollout_vllm_sampling_kwargs,
        ),
    )
    if load_initial_checkpoint:
      state = nnx.state(model)
      self._sampler.load_checkpoint(state)

  @property
  def mesh(self) -> jax.sharding.Mesh:
    return self._sampler.mesh

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the model."""
    kwargs.pop("internal_request_tags", None)
    self.output = self._sampler(
        input_strings=prompts,
        max_generation_steps=rollout_config.max_tokens_to_generate,
        max_prompt_length=rollout_config.max_prompt_length,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        seed=rollout_config.seed,
        echo=False,
        pad_output=True,
        **kwargs,
    )

    return base_rollout.RolloutOutput(
        text=self.output.text,
        logits=None,
        tokens=self.output.tokens,
        left_padded_prompt_tokens=self.output.padded_prompt_tokens,
        logprobs=self.output.logprobs,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Returns per-token log probabilities from the rollout policy."""
    # b/428730696, we cannot return self.output.logprobs yet
    # May need to validate if there will be any difference from recalculation
    return self.output.logprobs

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    self._sampler.update_params(params, filter_types)

  def pad_id(self) -> int:
    return self._sampler.tokenizer.pad_id()

  def eos_id(self) -> int:
    return self._sampler.tokenizer.eos_id()

  def model(self) -> nnx.Module:
    return self._sampler.transformer

  def close(self) -> None:
    self._sampler.stop()


class DistributedVllmRollout(base_rollout.BaseRollout):
  """Split-host wrapper that proxies rollout generation over a TCP transport."""

  def __init__(
      self,
      model: Any,
      tokenizer: Any,
      cache_config_or_size: base_rollout.CacheConfig | int,
      mesh: jax.sharding.Mesh,
      rollout_config: base_rollout.RolloutConfig,
      *,
      actor_owner_process_index: int,
      rollout_owner_process_index: int,
      process_hosts: list[str],
  ):
    self._model = model
    self._tokenizer = tokenizer
    self._cache_config_or_size = cache_config_or_size
    self._mesh = mesh
    self._rollout_config = rollout_config
    self._actor_owner_process_index = actor_owner_process_index
    self._rollout_owner_process_index = rollout_owner_process_index
    self._process_hosts = process_hosts
    self._request_counter = 0
    self.output: base_rollout.RolloutOutput | None = None
    self._local_rollout: VllmRollout | None = None
    self._listener: socket.socket | None = None
    self._listener_thread: threading.Thread | None = None
    self._shutdown_event = threading.Event()
    self._close_lock = threading.Lock()
    self._restart_lock = threading.Lock()
    self._closed = False
    self._latest_params_payload: list[tuple[Any, Any]] | None = None
    self._latest_filter_types: Optional[Tuple[Any, ...]] = None
    self._split_hosts = (
        self._actor_owner_process_index != self._rollout_owner_process_index
    )

    if jax.process_index() == self._rollout_owner_process_index:
      self._local_rollout = self._create_local_rollout(
          load_initial_checkpoint=not self._split_hosts
      )

    if (
        self._split_hosts
        and jax.process_index() == self._rollout_owner_process_index
    ):
      self.start_listener()

    if (
        self._split_hosts
        and jax.process_index() == self._actor_owner_process_index
    ):
      self.update_params(nnx.state(model))

  def start_listener(self) -> None:
    """Starts the split-host listener on the rollout-owner process."""
    if (
        not self._split_hosts
        or jax.process_index() != self._rollout_owner_process_index
    ):
      return

    if self._listener is not None:
      try:
        if (
            self._listener.fileno() >= 0
            and self._listener.getsockopt(
                socket.SOL_SOCKET, socket.SO_ACCEPTCONN
            )
        ):
          return
      except OSError:
        pass
      try:
        self._listener.close()
      except OSError:
        pass
      self._listener = None

    self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self._listener.bind(("", _host_transport_port()))
    self._listener.listen(128)
    self._listener.settimeout(None)
    logging.info(
        "Distributed rollout listener is ready on process %d at"
        " 0.0.0.0:%d (fd=%d); process_hosts=%s",
        jax.process_index(),
        _host_transport_port(),
        self._listener.fileno(),
        self._process_hosts,
    )
    self._listener_thread = threading.Thread(
        target=self._serve_messages,
        name="distributed-vllm-rollout-listener",
        daemon=True,
    )
    self._listener_thread.start()

  def _create_local_rollout(
      self, *, load_initial_checkpoint: bool
  ) -> VllmRollout:
    return VllmRollout(
        model=self._model,
        tokenizer=self._tokenizer,
        cache_config_or_size=self._cache_config_or_size,
        mesh=self._mesh,
        rollout_config=self._rollout_config,
        load_initial_checkpoint=load_initial_checkpoint,
    )

  def _is_driver_shutdown_error(self, exc: Exception) -> bool:
    return "Driver shut down" in str(exc)

  def _restart_local_rollout(
      self,
      *,
      params_payload: list[tuple[Any, Any]] | None = None,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    if jax.process_index() != self._rollout_owner_process_index:
      raise RuntimeError(
          "Only the rollout-owner process may restart the local vLLM rollout."
      )

    params_payload = params_payload or self._latest_params_payload
    filter_types = filter_types or self._latest_filter_types
    logging.warning(
        "Restarting local vLLM rollout on process %d after driver shutdown.",
        jax.process_index(),
    )

    if self._local_rollout is not None:
      try:
        self._local_rollout.close()
      except Exception:  # pragma: no cover - best-effort cleanup
        logging.exception("Failed to close stale local vLLM rollout cleanly.")

    self._local_rollout = self._create_local_rollout(load_initial_checkpoint=False)

    if params_payload is None:
      logging.warning(
          "No cached rollout params were available after restart; the local"
          " vLLM rollout will continue with its current weights."
      )
      return

    remote_params = _deserialize_state_from_transport(params_payload)
    self._local_rollout.update_params(remote_params, filter_types)

  @property
  def mesh(self) -> jax.sharding.Mesh | None:
    if self._local_rollout is None:
      return None
    return self._local_rollout.mesh

  def _next_request_id(
      self, internal_request_tags: Mapping[str, Any] | None
  ) -> str:
    self._request_counter += 1
    return _stable_request_id(internal_request_tags, self._request_counter)

  def _connect_to_process(self, process_index: int) -> socket.socket:
    last_error = None
    host = self._process_hosts[process_index]
    port = _host_transport_port()
    max_attempts = int(
        os.environ.get("TUNIX_DISTRIBUTED_ROLLOUT_CONNECT_ATTEMPTS", "300")
    )
    if max_attempts < 1:
      raise ValueError(
          "TUNIX_DISTRIBUTED_ROLLOUT_CONNECT_ATTEMPTS must be positive;"
          f" got {max_attempts}."
      )
    for attempt in range(1, max_attempts + 1):
      try:
        conn = socket.create_connection((host, port), 5.0)
        conn.settimeout(None)
        logging.info(
            "Connected to distributed rollout owner at %s:%d on attempt"
            " %d/%d.",
            host,
            port,
            attempt,
            max_attempts,
        )
        return conn
      except OSError as exc:  # pragma: no cover - runtime retry path
        last_error = exc
        if attempt == 1 or attempt % 15 == 0:
          logging.warning(
              "Waiting for distributed rollout owner at %s:%d (attempt"
              " %d/%d): %s",
              host,
              port,
              attempt,
              max_attempts,
              exc,
          )
        time.sleep(1.0)
    raise ConnectionError(
        "Failed to connect to rollout owner at"
        f" {host!r}:{port} after {max_attempts} attempts."
    ) from last_error

  def should_serve_only(self) -> bool:
    return (
        self._split_hosts
        and jax.process_index() == self._rollout_owner_process_index
    )

  def serve_until_shutdown(self) -> None:
    if not self.should_serve_only():
      return
    # Keep this idempotent check at the final service boundary as well. If a
    # later initialization cleanup invalidates the first post-GC listener,
    # recreate it before announcing service readiness.
    self.start_listener()
    logging.info(
        "Process %d entering distributed rollout service mode.",
        jax.process_index(),
    )
    self._shutdown_event.wait()

  def _send_message_to_rollout_owner(self, payload: dict[str, Any]) -> dict[str, Any]:
    with self._connect_to_process(self._rollout_owner_process_index) as conn:
      _send_message(conn, payload)
      return _recv_message(conn)

  def _send_generate_request_with_retry(
      self,
      payload: dict[str, Any],
      request_id: str,
  ) -> dict[str, Any]:
    last_error = None
    for attempt in range(2):
      try:
        response = self._send_message_to_rollout_owner(payload)
      except (ConnectionError, ConnectionResetError, OSError) as exc:
        last_error = exc
        if attempt == 0:
          logging.warning(
              "Distributed rollout request %s lost its connection to the"
              " rollout owner; retrying once after remote restart.",
              request_id,
          )
          time.sleep(1.0)
          continue
        raise

      if response.get("status") == "ok":
        return response

      error_text = str(response.get("error", ""))
      if attempt == 0 and "Driver shut down" in error_text:
        logging.warning(
            "Distributed rollout request %s hit a shut down vLLM driver on"
            " the rollout owner; retrying once.",
            request_id,
        )
        time.sleep(1.0)
        continue
      return response

    raise RuntimeError(
        "Distributed rollout request failed after retry:"
        f" request_id={request_id!r}, error={last_error!r}"
    )

  def _handle_message_connection(self, conn: socket.socket) -> None:
    with conn:
      message = _recv_message(conn)
      message_type = message.get("message_type")
      logging.info(
          "Accepted distributed rollout message %s of type %s on process %d.",
          message.get("request_id"),
          message_type,
          jax.process_index(),
      )

      if jax.process_index() != self._rollout_owner_process_index:
        raise RuntimeError(
            "Only the rollout-owner process may accept distributed rollout"
            " requests."
        )

      if self._local_rollout is None:
        raise RuntimeError("Rollout owner does not have a local rollout.")

      if message_type == "generate_request":
        request_id = message["request_id"]
        logging.info(
            "DISTRIBUTED_ROLLOUT_REQUEST phase=local_generate_start"
            " request_id=%s prompts=%d process=%d",
            request_id,
            len(message["prompts"]),
            jax.process_index(),
        )
        request_rollout = self._local_rollout
        try:
          try:
            output = request_rollout.generate(
                message["prompts"],
                message["rollout_config"],
                **message["kwargs"],
            )
          except RuntimeError as exc:
            if not self._is_driver_shutdown_error(exc):
              raise
            with self._restart_lock:
              if self._local_rollout is request_rollout:
                self._restart_local_rollout()
            output = self._local_rollout.generate(
                message["prompts"],
                message["rollout_config"],
                **message["kwargs"],
            )
        except Exception as exc:
          logging.exception(
              "Distributed rollout request %s failed during local generation.",
              request_id,
          )
          _send_message(
              conn,
              {
                  "status": "error",
                  "request_id": request_id,
                  "error": repr(exc),
              },
          )
          return
        logging.info(
            "DISTRIBUTED_ROLLOUT_REQUEST phase=local_generate_complete"
            " request_id=%s process=%d",
            request_id,
            jax.process_index(),
        )
        _send_message(
            conn,
            {
                "status": "ok",
                "request_id": request_id,
                "output": _serialize_rollout_output(output),
            },
        )
        return

      if message_type == "update_params":
        self._latest_params_payload = message["params"]
        self._latest_filter_types = message.get("filter_types")
        try:
          remote_params = _deserialize_state_from_transport(message["params"])
          try:
            self._local_rollout.update_params(
                remote_params, message.get("filter_types")
            )
          except RuntimeError as exc:
            if not self._is_driver_shutdown_error(exc):
              raise
            with self._restart_lock:
              self._restart_local_rollout(
                  params_payload=message["params"],
                  filter_types=message.get("filter_types"),
              )
        except Exception as exc:
          logging.exception("Distributed rollout weight sync failed.")
          _send_message(conn, {"status": "error", "error": repr(exc)})
          return
        _send_message(conn, {"status": "ok"})
        return

      if message_type == "shutdown":
        self._shutdown_event.set()
        _send_message(conn, {"status": "ok"})
        return

      _send_message(
          conn,
          {
              "status": "error",
              "error": (
                  "Unexpected distributed rollout message type:"
                  f" {message_type!r}"
              ),
          },
      )

  def _serve_messages(self) -> None:
    if self._listener is None:
      return
    while not self._shutdown_event.is_set():
      try:
        conn, _ = self._listener.accept()
      except OSError:
        if self._shutdown_event.is_set():
          return
        logging.exception(
            "Distributed rollout listener failed while accepting a connection"
            " on process %d (fd=%d).",
            jax.process_index(),
            self._listener.fileno(),
        )
        raise
      threading.Thread(
          target=self._handle_message_connection,
          args=(conn,),
          name="distributed-vllm-rollout-msg",
          daemon=True,
      ).start()

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the local or remote rollout worker."""
    internal_request_tags = kwargs.pop("internal_request_tags", None)
    request_id = self._next_request_id(internal_request_tags)

    if not self._split_hosts:
      if self._local_rollout is None:
        raise RuntimeError("Local vLLM rollout is not initialized.")
      self.output = self._local_rollout.generate(
          prompts, rollout_config, **kwargs
      )
      return self.output

    if jax.process_index() == self._actor_owner_process_index:
      logging.info(
          "Sending distributed rollout request %s with %d prompts from"
          " process %d to rollout owner %d.",
          request_id,
          len(prompts),
          jax.process_index(),
          self._rollout_owner_process_index,
      )
      response = self._send_generate_request_with_retry(
          {
              "message_type": "generate_request",
              "request_id": request_id,
              "prompts": prompts,
              "rollout_config": rollout_config,
              "kwargs": kwargs,
          },
          request_id,
      )
      if response.get("status") != "ok":
        raise RuntimeError(
            "Distributed rollout request failed on rollout owner:"
            f" {response!r}"
        )
      if response["request_id"] != request_id:
        raise RuntimeError(
            "Received mismatched distributed rollout response:"
            f" expected {request_id!r}, got {response['request_id']!r}."
        )
      self.output = _deserialize_rollout_output(response["output"])
      return self.output

    raise RuntimeError(
        "Distributed vLLM rollout requires the actor-owner process to issue"
        " requests while the rollout-owner process runs in service mode."
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    del prompt_tokens, completion_tokens, completion_mask
    if self.output is None:
      raise ValueError("Rollout output is not available yet.")
    return self.output.logprobs

  def update_params(
      self,
      params: jaxtyping.PyTree | None,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    if not self._split_hosts:
      if self._local_rollout is None:
        raise RuntimeError("Local vLLM rollout is not initialized.")
      self._local_rollout.update_params(params, filter_types)
      return

    if jax.process_index() == self._actor_owner_process_index:
      if params is None:
        raise ValueError("Actor owner requires params to sync rollout weights.")
      response = self._send_message_to_rollout_owner(
          {
              "message_type": "update_params",
              "params": _serialize_state_for_transport(params),
              "filter_types": filter_types,
          }
      )
      if response.get("status") != "ok":
        raise RuntimeError(
            "Distributed rollout weight sync failed:"
            f" {response!r}"
        )
      return
    if jax.process_index() == self._rollout_owner_process_index:
      return
    raise RuntimeError(
        "Distributed vLLM rollout weight sync requires actor-owner and"
        " rollout-owner processes."
    )

  def pad_id(self) -> int:
    return self._tokenizer.pad_id()

  def eos_id(self) -> int:
    return self._tokenizer.eos_id()

  def model(self) -> nnx.Module | None:
    if self._local_rollout is None:
      return None
    return self._local_rollout.model()

  def close(self) -> None:
    with self._close_lock:
      if self._closed:
        return
      self._closed = True

    self._shutdown_event.set()

    if self._split_hosts and jax.process_index() == self._actor_owner_process_index:
      try:
        response = self._send_message_to_rollout_owner(
            {"message_type": "shutdown"}
        )
        if response.get("status") != "ok":
          logging.warning(
              "Distributed rollout shutdown acknowledgement was not ok: %r",
              response,
          )
      except Exception:  # pragma: no cover - best-effort shutdown path
        logging.exception("Failed to shut down distributed rollout service.")

    if self._listener is not None:
      try:
        self._listener.close()
      except OSError:
        pass
      self._listener = None

    if self._local_rollout is not None:
      try:
        self._local_rollout.close()
      except Exception:  # pragma: no cover - best-effort shutdown path
        logging.exception("Failed to close the local vLLM rollout.")
      self._local_rollout = None
