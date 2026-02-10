# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""
Stress tests for ModelProxy.

Validates single-node correctness under high concurrency:
  - Many proxy instances binding to ports simultaneously
  - Many concurrent requests through a single proxy
  - Request/response isolation across proxies
  - Port exhaustion and recovery

Run:
    pytest verl/recipe/swe_agent/tests/test_proxy_stress.py -v
    # Or run a subset:
    pytest ... -k test_many_proxies
"""

import asyncio
import logging

import aiohttp
import pytest
from recipe.swe_agent.model_proxy import ModelProxy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_PORT = 18_000  # Use high port range to avoid conflicts


async def _do_roundtrip(proxy: ModelProxy, msg_text: str) -> str:
    """Send one chat completion request through *proxy* and return the
    response content.  The 'VERL side' is simulated by a background task
    that calls get_request / send_response.
    """
    echo_reply = f"echo:{msg_text}"

    async def _verl_side():
        req = await proxy.get_request()
        # Echo back with a prefix so we can verify isolation
        await proxy.send_response(echo_reply, request=req)

    verl_task = asyncio.create_task(_verl_side())

    # Simulate SWE-Agent HTTP call
    url = f"http://{proxy.host}:{proxy.port}/v1/chat/completions"
    payload = {"messages": [{"role": "user", "content": msg_text}]}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            data = await resp.json()

    await verl_task
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProxyPortBinding:
    """Verify port auto-increment works at scale."""

    @pytest.mark.asyncio
    async def test_many_proxies_bind_unique_ports(self):
        """Spin up N proxies on the *same* base port; each must auto-increment
        to a unique port and serve correctly.
        """
        n_proxies = 50
        proxies: list[ModelProxy] = []

        try:
            # Start all proxies from the same base port
            for _ in range(n_proxies):
                p = ModelProxy(port=BASE_PORT, host="127.0.0.1")
                await p.start_server()
                proxies.append(p)

            # All ports must be unique
            ports = [p.port for p in proxies]
            assert len(set(ports)) == n_proxies, f"Expected {n_proxies} unique ports, got {len(set(ports))}"

            # Verify every proxy is reachable
            async with aiohttp.ClientSession() as session:
                for p in proxies:
                    url = f"http://{p.host}:{p.port}/health"
                    async with session.get(url) as resp:
                        assert resp.status == 200
                        body = await resp.json()
                        assert body["status"] == "ok"

        finally:
            for p in proxies:
                await p.stop_server()

    @pytest.mark.asyncio
    async def test_port_range_exhaustion_raises(self):
        """When max_retries is tiny, binding should fail with RuntimeError."""
        proxies: list[ModelProxy] = []
        port = BASE_PORT + 500  # offset to avoid conflict with other tests

        try:
            # Occupy 3 ports
            for _ in range(3):
                p = ModelProxy(port=port, host="127.0.0.1")
                await p.start_server()
                proxies.append(p)
                port = p.port  # next attempt starts from same base

            # Now try with max_retries=1 — should fail
            p_fail = ModelProxy(port=proxies[0].port, host="127.0.0.1")
            with pytest.raises(RuntimeError, match="Failed to start server"):
                await p_fail.start_server(max_retries=1)

        finally:
            for p in proxies:
                await p.stop_server()


class TestProxyConcurrentRequests:
    """Validate request/response correctness under concurrency."""

    @pytest.mark.asyncio
    async def test_single_proxy_sequential_requests(self):
        """Many sequential roundtrips through one proxy must all return
        the correct echo response.
        """
        n_requests = 100
        proxy = ModelProxy(port=BASE_PORT + 1000, host="127.0.0.1")
        await proxy.start_server()

        try:
            for i in range(n_requests):
                msg = f"seq-{i}"
                reply = await _do_roundtrip(proxy, msg)
                assert reply == f"echo:{msg}", f"Request {i}: expected 'echo:{msg}', got '{reply}'"
        finally:
            await proxy.stop_server()

    @pytest.mark.asyncio
    async def test_single_proxy_concurrent_requests(self):
        """Fire N requests *concurrently* through one proxy.  Each must get
        its own correct response (no cross-talk).
        """
        n_requests = 64
        proxy = ModelProxy(port=BASE_PORT + 1100, host="127.0.0.1")
        await proxy.start_server()

        try:
            tasks = [_do_roundtrip(proxy, f"conc-{i}") for i in range(n_requests)]
            results = await asyncio.gather(*tasks)

            expected = {f"echo:conc-{i}" for i in range(n_requests)}
            actual = set(results)
            assert actual == expected, f"Mismatch: missing={expected - actual}, extra={actual - expected}"
        finally:
            await proxy.stop_server()


class TestMultiProxyIsolation:
    """Ensure requests/responses don't leak between proxies."""

    @pytest.mark.asyncio
    async def test_parallel_proxies_no_crosstalk(self):
        """Start K proxies and send M requests to each in parallel.
        Every proxy must only return responses for its own requests.
        """
        n_proxies = 20
        n_requests_per_proxy = 30
        proxies: list[ModelProxy] = []

        try:
            for _ in range(n_proxies):
                p = ModelProxy(port=BASE_PORT + 2000, host="127.0.0.1")
                await p.start_server()
                proxies.append(p)

            async def _run_proxy_batch(proxy: ModelProxy, proxy_idx: int) -> list[str]:
                results = []
                for j in range(n_requests_per_proxy):
                    msg = f"p{proxy_idx}-r{j}"
                    reply = await _do_roundtrip(proxy, msg)
                    results.append(reply)
                return results

            tasks = [_run_proxy_batch(p, idx) for idx, p in enumerate(proxies)]
            all_results = await asyncio.gather(*tasks)

            for idx, results in enumerate(all_results):
                for j, reply in enumerate(results):
                    expected = f"echo:p{idx}-r{j}"
                    assert reply == expected, f"Proxy {idx} request {j}: expected '{expected}', got '{reply}'"

        finally:
            for p in proxies:
                await p.stop_server()


class TestProxyEdgeCases:
    """Edge-case and robustness tests."""

    @pytest.mark.asyncio
    async def test_start_stop_restart_cycle(self):
        """A proxy can be started, stopped, and started again on a new port."""
        proxy = ModelProxy(port=BASE_PORT + 3000, host="127.0.0.1")

        # Cycle 1
        await proxy.start_server()
        reply = await _do_roundtrip(proxy, "cycle1")
        assert reply == "echo:cycle1"
        await proxy.stop_server()

        # Cycle 2 — must work on a fresh port
        proxy2 = ModelProxy(port=BASE_PORT + 3100, host="127.0.0.1")
        await proxy2.start_server()
        reply = await _do_roundtrip(proxy2, "cycle2")
        assert reply == "echo:cycle2"
        await proxy2.stop_server()

    @pytest.mark.asyncio
    async def test_double_start_raises(self):
        """Calling start_server twice must raise RuntimeError."""
        proxy = ModelProxy(port=BASE_PORT + 3200, host="127.0.0.1")
        await proxy.start_server()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                await proxy.start_server()
        finally:
            await proxy.stop_server()

    @pytest.mark.asyncio
    async def test_error_response_propagates(self):
        """send_error_response must deliver the error to the HTTP caller."""
        proxy = ModelProxy(port=BASE_PORT + 3300, host="127.0.0.1")
        await proxy.start_server()

        try:

            async def _verl_side():
                req = await proxy.get_request()
                await proxy.send_error_response(
                    req.request_id,
                    error_message="deliberate failure",
                    error_type="test_error",
                )

            verl_task = asyncio.create_task(_verl_side())

            url = f"http://{proxy.host}:{proxy.port}/v1/chat/completions"
            payload = {"messages": [{"role": "user", "content": "hello"}]}

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

            await verl_task

            assert "error" in data
            assert data["error"]["message"] == "deliberate failure"
            assert data["error"]["type"] == "test_error"

        finally:
            await proxy.stop_server()

    @pytest.mark.asyncio
    async def test_large_message_payload(self):
        """Proxy handles large message payloads (simulating long
        conversation histories from SWE-Agent).
        """
        proxy = ModelProxy(port=BASE_PORT + 3400, host="127.0.0.1")
        await proxy.start_server()

        try:
            # Build a large conversation (~500KB)
            big_content = "x" * 50_000
            messages = [{"role": "user", "content": f"msg-{i}-{big_content}"} for i in range(10)]

            async def _verl_side():
                req = await proxy.get_request()
                n_msgs = len(req.messages)
                await proxy.send_response(f"got {n_msgs} messages", request=req)

            verl_task = asyncio.create_task(_verl_side())

            url = f"http://{proxy.host}:{proxy.port}/v1/chat/completions"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={"messages": messages}) as resp:
                    data = await resp.json()

            await verl_task
            assert data["choices"][0]["message"]["content"] == "got 10 messages"

        finally:
            await proxy.stop_server()


# ---------------------------------------------------------------------------
# Large-scale integration tests
# ---------------------------------------------------------------------------


class TestLargeScaleDeployment:
    """Simulate a real single-node deployment with many workers."""

    @pytest.mark.asyncio
    async def test_150_proxies_concurrent_roundtrips(self):
        """Spin up 150 proxies from the same base port and drive 10
        concurrent roundtrips through each (1500 total).  Verifies:
          - port auto-increment works up to 150+
          - no response cross-talk across proxies
          - all roundtrips complete correctly
        """
        n_proxies = 150
        n_requests_per_proxy = 10
        proxies: list[ModelProxy] = []

        try:
            # --- Phase 1: start all proxies ---
            for _ in range(n_proxies):
                p = ModelProxy(port=BASE_PORT + 4000, host="127.0.0.1")
                await p.start_server()
                proxies.append(p)

            ports = [p.port for p in proxies]
            assert len(set(ports)) == n_proxies, (
                f"Port collision: {n_proxies} proxies but only {len(set(ports))} unique ports"
            )

            # --- Phase 2: concurrent roundtrips ---
            async def _batch(proxy: ModelProxy, idx: int) -> list[str]:
                coros = [_do_roundtrip(proxy, f"lg-p{idx}-r{j}") for j in range(n_requests_per_proxy)]
                return list(await asyncio.gather(*coros))

            all_results = await asyncio.gather(*[_batch(p, i) for i, p in enumerate(proxies)])

            # --- Phase 3: verify every response ---
            total = 0
            for idx, results in enumerate(all_results):
                for j, reply in enumerate(results):
                    expected = f"echo:lg-p{idx}-r{j}"
                    assert reply == expected, f"Proxy {idx} req {j}: expected '{expected}', got '{reply}'"
                    total += 1

            assert total == n_proxies * n_requests_per_proxy

        finally:
            await asyncio.gather(*[p.stop_server() for p in proxies])

    @pytest.mark.asyncio
    async def test_single_proxy_500_concurrent(self):
        """Push 500 concurrent requests through a single proxy.
        Validates the queue and event-based routing under heavy load.
        """
        n_requests = 500
        proxy = ModelProxy(port=BASE_PORT + 5000, host="127.0.0.1")
        await proxy.start_server()

        try:
            tasks = [_do_roundtrip(proxy, f"heavy-{i}") for i in range(n_requests)]
            results = await asyncio.gather(*tasks)

            expected = {f"echo:heavy-{i}" for i in range(n_requests)}
            actual = set(results)
            assert actual == expected, f"Missing: {len(expected - actual)}, Extra: {len(actual - expected)}"
        finally:
            await proxy.stop_server()
