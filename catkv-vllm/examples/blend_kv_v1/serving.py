#!/usr/bin/env python3
"""
Prometheus metrics collector for vLLM performance benchmarking.
Collects TTFT, TPOT, Throughput, and E2E Latency from Prometheus.
"""

import argparse
import json
import time
from datetime import datetime
from typing import Any, Dict, Optional

import requests


class PrometheusCollector:
    """Collector for Prometheus metrics from vLLM server."""

    def __init__(self, prometheus_host: str = "localhost", prometheus_port: int = 9090):
        """Initialize Prometheus collector.

        Args:
            prometheus_host: Prometheus server host
            prometheus_port: Prometheus server port (default: 9090)
        """
        self.prometheus_url = f"http://{prometheus_host}:{prometheus_port}"
        self.query_api = f"{self.prometheus_url}/api/v1/query"

    def query_prometheus(self, query: str) -> Optional[float]:
        """Query Prometheus and return scalar result."""
        try:
            params = {'query': query}
            response = requests.get(self.query_api, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data['status'] == 'success' and data['data']['result']:
                # Return the first result value
                return float(data['data']['result'][0]['value'][1])
            return None
        except Exception as e:
            print(f"Error querying Prometheus: {e}")
            return None

    def get_histogram_quantile(self, metric_name: str, quantile: float,
                               time_range: str = "5m", model_name: Optional[str] = None) -> Optional[float]:
        """Get quantile from histogram metric.

        Args:
            metric_name: Name of the histogram metric (without _bucket suffix)
            quantile: Quantile value (e.g., 0.5 for P50, 0.99 for P99)
            time_range: Time range for rate calculation
            model_name: Optional model name filter (note: requires model_name label in metrics)
        """
        # Note: model_name filtering only works if the vLLM server was started with model_name as a label
        # Most vLLM deployments don't have this label by default, so we'll ignore it if not needed
        model_filter = ''
        # if model_name:
        #     # Try with model_name first, but this might not exist
        #     print(f"Note: model_name filtering requested but may not be available in metrics")
        query = f'histogram_quantile({quantile}, sum by(le) (rate({metric_name}_bucket{{{model_filter}}}[{time_range}])))'
        return self.query_prometheus(query)

    def get_average(self, metric_name: str, time_range: str = "5m",
                   model_name: Optional[str] = None) -> Optional[float]:
        """Get average value from histogram metric.

        Args:
            metric_name: Name of the histogram metric (without _sum/_count suffix)
            time_range: Time range for rate calculation
            model_name: Optional model name filter (note: requires model_name label in metrics)
        """
        # Note: model_name filtering only works if the vLLM server was started with model_name as a label
        # Most vLLM deployments don't have this label by default, so we'll ignore it if not needed
        model_filter = ''
        if model_name:
            print(f"Note: model_name filtering requested but may not be available in metrics")
        query = f'''rate({metric_name}_sum{{{model_filter}}}[{time_range}])
/
rate({metric_name}_count{{{model_filter}}}[{time_range}])'''
        return self.query_prometheus(query)

    def get_rate(self, metric_name: str, time_range: str = "5m",
                model_name: Optional[str] = None) -> Optional[float]:
        """Get rate of a counter metric.

        Args:
            metric_name: Name of the counter metric
            time_range: Time range for rate calculation
            model_name: Optional model name filter (note: requires model_name label in metrics)
        """
        # Note: model_name filtering only works if the vLLM server was started with model_name as a label
        # Most vLLM deployments don't have this label by default, so we'll ignore it if not needed
        model_filter = ''
        if model_name:
            print(f"Note: model_name filtering requested but may not be available in metrics")
        query = f'rate({metric_name}{{{model_filter}}}[{time_range}])'
        return self.query_prometheus(query)

    def collect_vllm_metrics(self, time_range: str = "5m",
                            model_name: Optional[str] = None) -> Dict[str, Any]:
        """Collect all vLLM performance metrics.

        Args:
            time_range: Time range for metrics calculation
            model_name: Optional model name filter

        Returns:
            Dictionary containing TTFT, TPOT, Throughput, and E2E Latency metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'time_range': time_range,
            'model_name': model_name,
            'ttft': {},
            'tpot': {},
            'e2e_latency': {},
            'throughput': {}
        }

        # TTFT metrics
        print("Collecting TTFT metrics...")
        metrics['ttft']['p50'] = self.get_histogram_quantile('vllm:time_to_first_token_seconds', 0.5, time_range, model_name)
        metrics['ttft']['p90'] = self.get_histogram_quantile('vllm:time_to_first_token_seconds', 0.9, time_range, model_name)
        metrics['ttft']['p95'] = self.get_histogram_quantile('vllm:time_to_first_token_seconds', 0.95, time_range, model_name)
        metrics['ttft']['p99'] = self.get_histogram_quantile('vllm:time_to_first_token_seconds', 0.99, time_range, model_name)
        metrics['ttft']['average'] = self.get_average('vllm:time_to_first_token_seconds', time_range, model_name)

        # TPOT metrics
        print("Collecting TPOT metrics...")
        metrics['tpot']['p50'] = self.get_histogram_quantile('vllm:time_per_output_token_seconds', 0.5, time_range, model_name)
        metrics['tpot']['p90'] = self.get_histogram_quantile('vllm:time_per_output_token_seconds', 0.9, time_range, model_name)
        metrics['tpot']['p95'] = self.get_histogram_quantile('vllm:time_per_output_token_seconds', 0.95, time_range, model_name)
        metrics['tpot']['p99'] = self.get_histogram_quantile('vllm:time_per_output_token_seconds', 0.99, time_range, model_name)
        metrics['tpot']['average'] = self.get_average('vllm:time_per_output_token_seconds', time_range, model_name)

        # E2E Latency metrics
        print("Collecting E2E Latency metrics...")
        metrics['e2e_latency']['p50'] = self.get_histogram_quantile('vllm:e2e_request_latency_seconds', 0.5, time_range, model_name)
        metrics['e2e_latency']['p90'] = self.get_histogram_quantile('vllm:e2e_request_latency_seconds', 0.9, time_range, model_name)
        metrics['e2e_latency']['p95'] = self.get_histogram_quantile('vllm:e2e_request_latency_seconds', 0.95, time_range, model_name)
        metrics['e2e_latency']['p99'] = self.get_histogram_quantile('vllm:e2e_request_latency_seconds', 0.99, time_range, model_name)
        metrics['e2e_latency']['average'] = self.get_average('vllm:e2e_request_latency_seconds', time_range, model_name)

        # Throughput metrics
        print("Collecting Throughput metrics...")
        metrics['throughput']['prompt_tokens_per_sec'] = self.get_rate('vllm:prompt_tokens_total', time_range, model_name)
        metrics['throughput']['generation_tokens_per_sec'] = self.get_rate('vllm:generation_tokens_total', time_range, model_name)

        if metrics['throughput']['prompt_tokens_per_sec'] and metrics['throughput']['generation_tokens_per_sec']:
            metrics['throughput']['total_tokens_per_sec'] = (
                metrics['throughput']['prompt_tokens_per_sec'] +
                metrics['throughput']['generation_tokens_per_sec']
            )

        return metrics

    def monitor_during_test(self, duration: int = 60, interval: int = 10,
                          model_name: Optional[str] = None) -> Dict[str, Any]:
        """Monitor metrics during a test for specified duration.

        Args:
            duration: Total monitoring duration in seconds
            interval: Sampling interval in seconds
            model_name: Optional model name filter

        Returns:
            Dictionary with samples and final metrics
        """
        samples = []
        start_time = time.time()
        end_time = start_time + duration

        print(f"Starting monitoring for {duration} seconds...")

        while time.time() < end_time:
            elapsed = int(time.time() - start_time)
            print(f"\n[{elapsed}s/{duration}s] Collecting metrics sample...")

            metrics = self.collect_vllm_metrics(time_range='1m', model_name=model_name)
            metrics['elapsed_time'] = elapsed
            samples.append(metrics)

            if time.time() < end_time:
                time.sleep(interval)

        # Collect final metrics with full time range
        print(f"\nCollecting final metrics over full {duration}s duration...")
        final_metrics = self.collect_vllm_metrics(
            time_range=f'{duration}s',
            model_name=model_name
        )

        return {
            'test_duration': duration,
            'sampling_interval': interval,
            'num_samples': len(samples),
            'samples': samples,
            'final_metrics': final_metrics
        }


def main():
    parser = argparse.ArgumentParser(
        description="Collect vLLM performance metrics from Prometheus"
    )
    parser.add_argument(
        "--prometheus-host",
        type=str,
        default="localhost",
        help="Prometheus server host (default: localhost)"
    )
    parser.add_argument(
        "--prometheus-port",
        type=int,
        default=9091,  # Changed default to 9091 to match docker-compose mapping
        help="Prometheus server port (default: 9091 for docker-compose setup)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Filter metrics by model name (optional)"
    )
    parser.add_argument(
        "--time-range",
        type=str,
        default="5m",
        help="Time range for metrics calculation (default: 5m)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="prometheus_metrics.json",
        help="Output JSON file (default: prometheus_metrics.json)"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor metrics continuously during test"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Monitoring duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Sampling interval in seconds (default: 10)"
    )

    args = parser.parse_args()

    collector = PrometheusCollector(
        prometheus_host=args.prometheus_host,
        prometheus_port=args.prometheus_port
    )

    if args.monitor:
        data = collector.monitor_during_test(
            duration=args.duration,
            interval=args.interval,
            model_name=args.model_name
        )
    else:
        print(f"Collecting metrics for time range: {args.time_range}")
        data = collector.collect_vllm_metrics(
            time_range=args.time_range,
            model_name=args.model_name
        )

    # Save to file
    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nMetrics saved to {args.output}")

    # Print summary
    if not args.monitor:
        print("\n=== Metrics Summary ===")
        if data.get('ttft'):
            print("\nTTFT (Time To First Token):")
            for k, v in data['ttft'].items():
                if v is not None:
                    print(f"  {k}: {v:.4f}s")
        if data.get('tpot'):
            print("\nTPOT (Time Per Output Token):")
            for k, v in data['tpot'].items():
                if v is not None:
                    print(f"  {k}: {v:.4f}s")
        if data.get('e2e_latency'):
            print("\nE2E Latency:")
            for k, v in data['e2e_latency'].items():
                if v is not None:
                    print(f"  {k}: {v:.4f}s")
        if data.get('throughput'):
            print("\nThroughput:")
            for k, v in data['throughput'].items():
                if v is not None:
                    print(f"  {k}: {v:.2f} tokens/sec")


if __name__ == "__main__":
    main()